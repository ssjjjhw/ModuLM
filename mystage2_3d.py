import datetime
import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.mystage2_dm_3d import Stage2DM,Stage2DM_double,Stage2DM_double_value,Stage2DM_double_DDIvalue,Stage2DM_double_fgtvalue,Stage2DM_universal
from data_provider.iupac_dm import IupacDM
from data_provider.stage2_chebi_dm import Stage2CheBIDM
from model.my_blip2_stage2_3d import Blip2Stage2
import torch.distributed as dist
from model.unimol import SimpleUniMolModel
# from pytorch_lightning.strategies import FSDPStrategy
# torch.set_default_dtype(torch.float16)
# os.environ["TOKENIZERS_PARALLELISM"] = "True"

os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["NCCL_TIMEOUT"] = "36000000"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"


## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


# class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
class MyDDPSpawnStrategy(strategies.DDPStrategy):
# class MyDDPSpawnStrategy(strategies.SingleDeviceStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)
    def _setup(self, trainer, model):
        # 设置 NCCL 超时时间
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(hours=8)
        )
        super()._setup(trainer, model)

def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args,map_location="cpu")
        
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.opt_model.find('galactica') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    if args.iupac_prediction:
        dm = IupacDM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)
    else:
        if args.root.lower().find('chebi') >= 0:
            dm = Stage2CheBIDM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)
        else:
            if args.double == True:
                dm = Stage2DM_double(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len,model.blip2opt.dictionary, tokenizer,False ,args)
            else:
                dm = Stage2DM_universal(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)
    
    callbacks = []
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
    #                                      filename='{epoch:02d}', 
    #                                      every_n_epochs=args.save_every_n_epochs, 
    #                                      save_last=True, 
    #                                      save_top_k=-1))
    callbacks.append(plc.ModelCheckpoint(dirpath="/home/cz/MolTC-main/"+args.filename+"/", 
                                        filename='{epoch:02d}', 
                                        every_n_epochs=args.save_every_n_epochs, 
                                        save_last=True, 
                                        save_top_k=-1))
    
    deepspeed_config = {
    # "train_batch_size": 32,
    # "gradient_accumulation_steps": 1,
    "fp16": {  # 禁用 FP16
        "enabled": False
    },
    "bf16": {  # 禁用 BF16
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3
    },
    "zero_allow_untested_optimizer": True
    }

    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(config=deepspeed_config)
        else:
            # print('&&&&&&&&&')
            strategy = MyDDPSpawnStrategy(find_unused_parameters=True)
    else:
        print('kkkkkkkkkkkkkkkkk')
        strategy = None
        args.devices = eval(args.devices)
        print(args.devices)
        # args.devices = torch.device("cuda:1")
        # args.device = "1"
    logger = CSVLogger(save_dir=f'/home/cz/MolTC-main/{args.filename}/')
        #if name.startswith('fc_layer'):
            #param.requires_grad = False
    
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger,
                                         precision="bf16",
                                         #控制单卡训练设置args中的device同时修改此处
                                        #  gpus = [7],
                                        #  limit_train_batches=100,
                                         )
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--iupac_prediction', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2DM.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--double', type=bool, default=True)
    parser.add_argument('--solve', type=bool, default=False)
    parser.add_argument('--valid_root', type=str, default='data/PubChemDataset_v4')
    parser.add_argument('--DDI', type=bool, default=False)
    parser.add_argument('--fangguangtuan', type=bool, default=False)


    parser.add_argument('--zijiegou', type=bool, default=False)
    parser.add_argument('--autozijiegou', type=bool, default=False)




    parser.add_argument('--use_2d', type=bool, default=False)
    parser.add_argument('--use_3d', type=bool, default=False)
    parser.add_argument('--use_inter', type=bool, default=False)

    parser.add_argument('--backbone', type=str, default='galactica-1.3b')
    parser.add_argument('--alignment', type=str, default='mlp')
    parser.add_argument('--graph2d', type=str, default='gin')
    parser.add_argument('--graph3d', type=str, default='unimol')
    parser.add_argument('--interaction', type=str, default='mean')

    parser = SimpleUniMolModel.add_args(parser)

    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args






if __name__ == '__main__':




    args = get_args()
    print(args)
    main(args)

