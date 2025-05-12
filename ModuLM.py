from mystage2_3d import *




class ModuLMConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self._add_general_args()
        self._add_model_args()
        self._add_data_args()
        self._add_trainer_args()
        self._add_graph_args()

    def _add_general_args(self):
        self.parser.add_argument('--filename', type=str, default="stage2_test")
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--mode', type=str, default='ft')
        self.parser.add_argument('--strategy_name', type=str, default=None)
        self.parser.add_argument('--ckpt_path', type=str, default=None)

    def _add_model_args(self):
        self.parser.add_argument('--init_checkpoint', type=str, default='all_checkpoints/pretrain1/last.ckpt')
        self.parser.add_argument('--stage1_path', type=str, default=None)
        self.parser.add_argument('--stage2_path', type=str, default='all_checkpoints/stage2/last.ckpt')
        self.parser.add_argument('--opt_model', type=str, default='galactica-1.3b')
        self.parser.add_argument('--backbone', type=str, default='galactica-1.3b')
        self.parser.add_argument('--alignment', type=str, default='mlp')
        self.parser.add_argument('--interaction', type=str, default='mean')

    def _add_data_args(self):
        self.parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        self.parser.add_argument('--valid_root', type=str, default='data/PubChemDataset_v4')
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--inference_batch_size', type=int, default=2)
        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument('--text_max_len', type=int, default=128)
        self.parser.add_argument('--iupac_prediction', action='store_true')
        self.parser.add_argument('--double', type=bool, default=True)
        self.parser.add_argument('--solve', type=bool, default=False)
        self.parser.add_argument('--DDI', type=bool, default=False)
        self.parser.add_argument('--fangguangtuan', type=bool, default=False)
        self.parser.add_argument('--zijiegou', type=bool, default=False)
        self.parser.add_argument('--autozijiegou', type=bool, default=False)

    def _add_graph_args(self):
        self.parser.add_argument('--graph2d', type=str, default='gin')
        self.parser.add_argument('--graph3d', type=str, default='unimol')
        self.parser.add_argument('--use_2d', type=bool, default=False)
        self.parser.add_argument('--use_3d', type=bool, default=False)
        self.parser.add_argument('--use_inter', type=bool, default=False)

    def _add_trainer_args(self):
        self.parser.add_argument('--accelerator', type=str, default='gpu')
        self.parser.add_argument('--devices', type=str, default='0,1,2,3')
        self.parser.add_argument('--precision', type=str, default='bf16')
        self.parser.add_argument('--max_epochs', type=int, default=10)
        self.parser.add_argument('--accumulate_grad_batches', type=int, default=1)
        self.parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
        self.parser.add_argument('--save_every_n_epochs', type=int, default=100)
        self.parser.add_argument('--caption_eval_epochs', type=int, default=5)

        self.parser.add_argument('--num_query_token', type=int, default=10)
        self.parser.add_argument('--min_len', type=int, default=10)
        self.parser.add_argument('--max_len', type=int, default=30)

    def parse(self):
        return self.parser.parse_args()
    
    def parse_from_json(self, json_path):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        args_list = []
        for key, value in config_dict.items():
            if isinstance(value, bool):
                if value:  # 对于布尔值，只添加 True，False 使用默认值
                    args_list.append(f'--{key}')
            else:
                args_list.append(f'--{key}')
                args_list.append(str(value))

        return self.parser.parse_args(args_list)








class ModuLM:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.dm = None
        pl.seed_everything(self.args.seed)

    def load_model(self):
        if self.args.init_checkpoint:
            self.model = Blip2Stage2.load_from_checkpoint(
                self.args.init_checkpoint, strict=False, args=self.args, map_location="cpu"
            )
            print(f"Loaded init checkpoint from {self.args.init_checkpoint}")
        elif self.args.stage2_path:
            self.model = Blip2Stage2(self.args)
            ckpt = torch.load(self.args.stage2_path, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"Loaded stage2 model from {self.args.stage2_path}")
        elif self.args.stage1_path:
            self.model = Blip2Stage2(self.args)
            self.model.load_from_stage1_checkpoint(self.args.stage1_path)
            print(f"Loaded stage1 model from {self.args.stage1_path}")
        else:
            self.model = Blip2Stage2(self.args)
            print("Initialized model from scratch")
        
        print('Total params:', sum(p.numel() for p in self.model.parameters()))
        return self.model

    def load_data(self):
        if self.model is None:
            raise ValueError("Model must be loaded before loading data.")
        
        if self.args.opt_model.find('galactica') >= 0:
            tokenizer = self.model.blip2opt.opt_tokenizer
        elif self.args.opt_model.find('llama') >= 0 or self.args.opt_model.find('vicuna') >= 0:
            tokenizer = self.model.blip2opt.llm_tokenizer
        else:
            raise NotImplementedError

        if self.args.iupac_prediction:
            self.dm = IupacDM(self.args.mode, self.args.num_workers, self.args.batch_size, self.args.root,
                              self.args.text_max_len, tokenizer, self.args)
        elif self.args.root.lower().find('chebi') >= 0:
            self.dm = Stage2CheBIDM(self.args.mode, self.args.num_workers, self.args.batch_size, self.args.root,
                                    self.args.text_max_len, tokenizer, self.args)
        elif self.args.double:
            self.dm = Stage2DM_double(self.args.mode, self.args.num_workers, self.args.batch_size, self.args.root,
                                      self.args.text_max_len, self.model.blip2opt.dictionary, tokenizer, False, self.args)
        else:
            self.dm = Stage2DM_universal(self.args.mode, self.args.num_workers, self.args.batch_size, self.args.root,
                                         self.args.text_max_len, tokenizer, self.args)
        return self.dm

    def train(self):
        if self.model is None or self.dm is None:
            raise ValueError("Both model and data must be loaded before training.")

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=f"/home/cz/MolTC-main/{self.args.filename}/",
                filename="{epoch:02d}",
                every_n_epochs=self.args.save_every_n_epochs,
                save_last=True,
                save_top_k=-1
            )
        ]

        if len(self.args.devices.split(',')) > 1:
            if self.args.strategy_name == 'fsdp':
                strategy = pl.strategies.DDPFullyShardedNativeStrategy()
            elif self.args.strategy_name == 'deepspeed':
                strategy = pl.strategies.DeepSpeedStrategy(config={
                    "bf16": {"enabled": True},
                    "fp16": {"enabled": False},
                    "zero_optimization": {"stage": 3},
                    "zero_allow_untested_optimizer": True
                })
            else:
                strategy = MyDDPSpawnStrategy(find_unused_parameters=True)
        else:
            strategy = None
            self.args.devices = eval(self.args.devices)

        logger = CSVLogger(save_dir=f"/home/cz/MolTC-main/{self.args.filename}/")

        trainer = Trainer.from_argparse_args(self.args,
                                             callbacks=callbacks,
                                             strategy=strategy,
                                             logger=logger,
                                             precision="bf16")

        if self.args.mode in {"pretrain", "ft"}:
            trainer.fit(self.model, datamodule=self.dm, ckpt_path=self.args.ckpt_path)
        elif self.args.mode == "eval":
            trainer.fit_loop.epoch_progress.current.completed = self.args.caption_eval_epoch - 1
            trainer.validate(self.model, datamodule=self.dm)
        else:
            raise NotImplementedError()

    def run(self):
        self.load_model()
        self.load_data()
        self.train()


