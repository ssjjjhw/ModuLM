# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_caption_dataset import MoleculeCaption, MoleculeCaption_double,MoleculeCaption_double_value,MoleculeCaption_double_DDIvalue,MoleculeCaption_double_fgtvalue,MoleculeCaption_universal
import re

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    
    # text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        
        ## deal with prompt
        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]
        # prompt_tokens = self.tokenizer(smiles_prompt, return_tensors='pt', max_length=self.text_max_len, padding='longest', truncation=True, return_attention_mask=True)
        # prompt_lens = prompt_tokens.attention_mask.sum(dim=1)

        # smiles_prompt = [p) for p in smiles_prompt]
        ## concate text and prompt

        # texts = [escape_custom_split_sequence(prompt + text) for prompt, text in zip(smiles_prompt, texts)]
        #print(smiles_prompt)
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        #在这里是真正转化成tokens
        return graphs, smiles_prompt_tokens, text_tokens


    
class TrainCollater_double:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id,zijiegou=False):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.zijiegou = zijiegou
    def __call__(self, batch):
        if self.zijiegou==False:
            #就应该是在这里完成版本更替。
            #在这里把两个smiles拼在一起，并且完成is_mol的两端修改，然后
            graphs1,graphs2, texts ,smiles_prompt= zip(*batch)
            # print(smiles_prompt[][0],'!!!!!!!!!!!')
            # print(smiles_prompt[1],'!!!!!!!!!!!1')
            # print(smiles_prompt[2],'!!!!!!!!!!!2')
            # print(smiles_prompt[3],'!!!!!!!!!!!3')
            # print('111111',smiles_prompt[0],'&&&&&&&&&&&',smiles_prompt)
            # print(texts)#Drug1 and #Drug2 do not exhibit drug-drug interaction.'
            graphs1 = self.collater(graphs1)
            ## deal with prompt
            
            smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*8+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*8+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]
            # smiles_prompt = ['<GraEmb1>'*25+'</s>'+'<GraEmb2>'*25+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]



            smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                                truncation=False,
                                                padding='longest',
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                return_attention_mask=True)


            is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
            smiles_prompt_tokens['is_mol_token'] = is_mol_token

            is_GraEmb1_token = smiles_prompt_tokens.input_ids == self.tokenizer.GraEmb1_token_id
            smiles_prompt_tokens['is_GraEmb1_token'] = is_GraEmb1_token

            is_GraEmb2_token = smiles_prompt_tokens.input_ids == self.tokenizer.GraEmb2_token_id
            smiles_prompt_tokens['is_GraEmb2_token'] = is_GraEmb2_token

            #print(texts1)
            text_tokens = self.tokenizer(text=texts,
                                        truncation=True,
                                        padding='longest',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
            #这里的attentionmask只对padding的部分进行了
            # print('1111111111',texts[0],'222222222222',text_tokens[0],'3333333333333',text_tokens.attention_mask[0])
            #1111111111 #Drug1 and #Drug2 exhibit drug-drug interaction. 
            #222222222222 Encoding(num_tokens=18, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]) 
            #3333333333333 tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
            is_start_token = text_tokens.input_ids == self.tokenizer.start_property_token_id
            text_tokens['is_start_property'] = is_start_token

            is_end_token = text_tokens.input_ids == self.tokenizer.end_property_token_id
            text_tokens['is_end_property'] = is_end_token
            
            graphs2 = self.collater(graphs2)
            
        
            #在这里是真正转化成tokens
            return graphs1,graphs2,smiles_prompt_tokens  ,text_tokens
        else:
            #就应该是在这里完成版本更替。
            #在这里把两个smiles拼在一起，并且完成is_mol的两端修改，然后
            graphs1,graphs2, texts ,smiles_prompt,zijiegou1,zijiegou2= zip(*batch)
            # print('111111',smiles_prompt[0],'&&&&&&&&&&&',smiles_prompt)
            # print(texts)#Drug1 and #Drug2 do not exhibit drug-drug interaction.'
            graphs1 = self.collater(graphs1)
            ## deal with prompt
            # print('111111111111111',smiles_prompt[0])</s> [START_I_SMILES]CCC1=NN(CCCN2CCN(CC2)C2=CC(Cl)=CC=C2)C(=O)N1CCOC1=CC=CC=C1[END_I_SMILES].</s> 
            smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]
            # print('************',smiles_prompt[0])对原smiles添加了mol*n个标记作为结尾，并在每个原子两旁插入分隔符

            smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                                truncation=False,
                                                padding='longest',
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                return_attention_mask=True)
            # print('1111111111',smiles_prompt[0],'222222222222',smiles_prompt_tokens[0],'3333333333333',smiles_prompt_tokens.attention_mask[0])
            #111是插入特殊分隔符后的smiles字符串
            #333是对padding后的字符串进行标记，主要是padding
            
            is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
            smiles_prompt_tokens['is_mol_token'] = is_mol_token

            #print(texts1)
            text_tokens = self.tokenizer(text=texts,
                                        truncation=True,
                                        padding='longest',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
            #这里的attentionmask只对padding的部分进行了
            # print('1111111111',texts[0],'222222222222',text_tokens[0],'3333333333333',text_tokens.attention_mask[0])
            #1111111111 #Drug1 and #Drug2 exhibit drug-drug interaction. 
            #222222222222 Encoding(num_tokens=18, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]) 
            #3333333333333 tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
            
            
            graphs2 = self.collater(graphs2)
            
        
            #在这里是真正转化成tokens
            return graphs1,graphs2,zijiegou1,zijiegou2,smiles_prompt_tokens  ,text_tokens,
    
    
    
    

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]

        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                    #    max_length=self.text_max_len, 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        return graphs, smiles_prompt_tokens, texts
    

class InferenceCollater_double:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id,zijiegou=False):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.zijiegou = zijiegou
        
    def __call__(self, batch):
        if self.zijiegou==False:
            graphs1,graphs2, texts ,smiles_prompt = zip(*batch)
            graphs1 = self.collater(graphs1)
            graphs2 = self.collater(graphs2)
            smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*8+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*8+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]
            # smiles_prompt = ['<GraEmb1>'*25+'</s>'+'<GraEmb2>'*25+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]

            ## deal with prompt
            smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                        return_tensors='pt', 
                                        #    max_length=self.text_max_len, 
                                        padding='longest', 
                                        truncation=False, 
                                        return_attention_mask=True)

            is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
            smiles_prompt_tokens['is_mol_token'] = is_mol_token

            is_GraEmb1_token = smiles_prompt_tokens.input_ids == self.tokenizer.GraEmb1_token_id
            smiles_prompt_tokens['is_GraEmb1_token'] = is_GraEmb1_token

            is_GraEmb2_token = smiles_prompt_tokens.input_ids == self.tokenizer.GraEmb2_token_id
            smiles_prompt_tokens['is_GraEmb2_token'] = is_GraEmb2_token
            
            return graphs1,graphs2, smiles_prompt_tokens, texts       
        else:
            graphs1,graphs2, texts ,smiles_prompt,zijiegou1,zijiegou2 = zip(*batch)
            graphs1 = self.collater(graphs1)
            graphs2 = self.collater(graphs2)
            smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]

            ## deal with prompt
            smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                        return_tensors='pt', 
                                        #    max_length=self.text_max_len, 
                                        padding='longest', 
                                        truncation=False, 
                                        return_attention_mask=True)

            is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
            smiles_prompt_tokens['is_mol_token'] = is_mol_token
            return graphs1,graphs2,zijiegou1,zijiegou2,smiles_prompt_tokens ,texts
    
class Stage2DM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        # root: str = 'data/',
        root: str = '/home/amax/zjh/MolTC-main/data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset = MoleculeCaption('qformer_data/train/', text_max_len, self.prompt)
        self.train_dataset = MoleculeCaption('qformer_data/train/', text_max_len, self.prompt)
        self.val_dataset = MoleculeCaption('qformer_data/val/', text_max_len, self.prompt)
        self.test_dataset = MoleculeCaption('qformer_data/val/', text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
class Stage2DM_double(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        # root: str = 'data/',
        root: str = '/home/cz/MolTC-main/data/',
        text_max_len: int = 128,
        tokenizer=None,
        zijiegou=False,
        args=None,
        
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        
        self.zijiegou = zijiegou
        
        if zijiegou==True:
            self.pretrain_dataset =  MoleculeCaption_double(root, text_max_len, self.prompt,True)
            self.train_dataset =  MoleculeCaption_double(root, text_max_len, self.prompt,True)
            # self.val_dataset =  MoleculeCaption_double('qformer_data/val/', text_max_len, self.prompt)
            self.val_dataset =  MoleculeCaption_double(args.valid_root, text_max_len, self.prompt,True)
            self.test_dataset = MoleculeCaption_double(args.valid_root, text_max_len, self.prompt,True)
        else:
            # print('****************')
            self.pretrain_dataset =  MoleculeCaption_double(root, text_max_len, self.prompt,False,True,self.args.solve)
            self.train_dataset =  MoleculeCaption_double(root, text_max_len, self.prompt,False,True,self.args.solve)
            # self.val_dataset =  MoleculeCaption_double('qformer_data/val/', text_max_len, self.prompt)
            self.val_dataset =  MoleculeCaption_double(args.valid_root, text_max_len, self.prompt,False,True,self.args.solve)
            self.test_dataset = MoleculeCaption_double(args.valid_root, text_max_len, self.prompt,False,True,self.args.solve)
        
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token*2
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.zijiegou),
            )
        else:
            raise NotImplementedError
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.zijiegou),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            #注意 在这里和下面我把测试集数据也都改了
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.zijiegou),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.zijiegou),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
class Stage2DM_double_value(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeCaption_double_value(root, text_max_len, self.prompt)
        self.train_dataset =  MoleculeCaption_double_value(root, text_max_len, self.prompt)
        # % val 和test 一样,这里没有验证机,只划分了训练和测试;不过代码中都写的验证集(就是测试集)
        self.val_dataset =  MoleculeCaption_double_value(args.valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeCaption_double_value(args.valid_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            #注意 在这里和下面我把测试集数据也都改了
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
    
    
    
    
class Stage2DM_double_DDIvalue(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeCaption_double_DDIvalue(root, text_max_len, self.prompt)
        self.train_dataset =  MoleculeCaption_double_DDIvalue(root, text_max_len, self.prompt)
        self.val_dataset =  MoleculeCaption_double_DDIvalue(self.args.valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeCaption_double_DDIvalue(self.args.valid_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            print("*******************************************")
            print(self.args.valid_root)
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            #注意 在这里和下面我把测试集数据也都改了
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser

class Stage2DM_double_fgtvalue(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeCaption_double_fgtvalue(root, text_max_len, self.prompt)
        self.train_dataset =  MoleculeCaption_double_fgtvalue(root, text_max_len, self.prompt)
        self.val_dataset =  MoleculeCaption_double_fgtvalue(valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeCaption_double_fgtvalue(valid_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            #注意 在这里和下面我把测试集数据也都改了
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser



class Stage2DM_universal(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeCaption_universal(root, text_max_len, self.prompt)
        self.train_dataset =  MoleculeCaption_universal(root, text_max_len, self.prompt)
        self.val_dataset =  MoleculeCaption_universal(self.args.valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeCaption_universal(self.args.valid_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            print(self.pretrain_dataset)
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            #注意 在这里和下面我把测试集数据也都改了
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
