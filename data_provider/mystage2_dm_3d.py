# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_caption_dataset3d import MoleculeCaption, MoleculeCaption_double,MoleculeCaption_double_value,MoleculeCaption_double_DDIvalue,MoleculeCaption_double_fgtvalue,MoleculeCaption_universal
import re
from unicore.data import data_utils

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


def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class D3Collater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple
    
    def __call__(self, samples):
        atom_vec, coordinates, edge_type, dist, smiles = zip(*samples)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_coordinates = collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        return padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles


    
class TrainCollater_double:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id,pad_idx,zijiegou=False,args=None):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        ###构象
        self.d3_collater = D3Collater(pad_idx)
        self.args = args
        ###
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.zijiegou = zijiegou
    def __call__(self, batch):
        graphs1,graphs2, texts ,smiles_prompt= zip(*batch)
        
        #graphs1 batch_size * 20 *needed
        #graph_batch1[0]对应的0分子的20个构象，输入给模型直接得到该分子的20个构象信息
        if self.args.use_3d==True:
            graph_batch1 = []
            for i in range(len(graphs1)):
                padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs1[i])
                graph_batch1.append((padded_atom_vec, padded_coordinates, padded_dist, padded_edge_type))
            
            graph_batch2=[]
            for i in range(len(graphs2)):
                padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs2[i])
                graph_batch2.append((padded_atom_vec, padded_coordinates, padded_dist, padded_edge_type))

        elif self.args.use_2d==True:
            graph_batch1 = self.collater(graphs1)
            graph_batch2 = self.collater(graphs2)
            
        
        if self.args.use_inter==True:
            smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*self.args.num_query_token+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*self.args.num_query_token+'</s>'+'Their interaction information are as follows:'+'<mol>'*self.args.num_query_token+p[3] for p in smiles_prompt]
        else:
            smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*self.args.num_query_token+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*self.args.num_query_token+'</s>'+p[3] for p in smiles_prompt]
        # print(smiles_prompt)
        # smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]

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

        is_start_token = text_tokens.input_ids == self.tokenizer.start_property_token_id
        text_tokens['is_start_property'] = is_start_token

        is_end_token = text_tokens.input_ids == self.tokenizer.end_property_token_id
        text_tokens['is_end_property'] = is_end_token
        
        
        return graph_batch1,graph_batch2,smiles_prompt_tokens,text_tokens
    
    
    

class InferenceCollater_double:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id,pad_idx,zijiegou=False,args=None):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.d3_collater = D3Collater(pad_idx)
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.zijiegou = zijiegou
        self.args=args
    def __call__(self, batch):
        
        # graphs1,graphs2, texts ,smiles_prompt = zip(*batch)

        # graph_batch1 = []
        # for i in range(len(graphs1)):
        #     padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs1[i])
        #     graph_batch1.append((padded_atom_vec, padded_dist, padded_edge_type))
        
        # graph_batch2=[]
        # for i in range(len(graphs2)):
        #     padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs2[i])
        #     graph_batch2.append((padded_atom_vec, padded_dist, padded_edge_type))
    
        # smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*self.args.num_query_token+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*self.args.num_query_token+'</s>'+p[3] for p in smiles_prompt]
        # # smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]

        # smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
        #                             return_tensors='pt', 
        #                             #    max_length=self.text_max_len, 
        #                             padding='longest', 
        #                             truncation=False, 
        #                             return_attention_mask=True)

        # is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        # smiles_prompt_tokens['is_mol_token'] = is_mol_token

        # is_GraEmb1_token = smiles_prompt_tokens.input_ids == self.tokenizer.GraEmb1_token_id
        # smiles_prompt_tokens['is_GraEmb1_token'] = is_GraEmb1_token

        # is_GraEmb2_token = smiles_prompt_tokens.input_ids == self.tokenizer.GraEmb2_token_id
        # smiles_prompt_tokens['is_GraEmb2_token'] = is_GraEmb2_token
        
        # return graph_batch1,graph_batch2, smiles_prompt_tokens, texts     
        graphs1,graphs2, texts ,smiles_prompt= zip(*batch)
        
        #graphs1 batch_size * 20 *needed
        #graph_batch1[0]对应的0分子的20个构象，输入给模型直接得到该分子的20个构象信息
        if self.args.use_3d==True:
            graph_batch1 = []
            for i in range(len(graphs1)):
                padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs1[i])
                graph_batch1.append((padded_atom_vec, padded_coordinates, padded_dist, padded_edge_type))
            
            graph_batch2=[]
            for i in range(len(graphs2)):
                padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs2[i])
                graph_batch2.append((padded_atom_vec,padded_coordinates, padded_dist, padded_edge_type))

        elif self.args.use_2d==True:
            graph_batch1 = self.collater(graphs1)
            graph_batch2 = self.collater(graphs2)
            
        if self.args.use_inter==True:
            smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*self.args.num_query_token+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*self.args.num_query_token+'</s>'+'Their interaction information are as follows:'+'<mol>'*self.args.num_query_token+p[3] for p in smiles_prompt]
        else:
            smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'<GraEmb1>'*self.args.num_query_token+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'<GraEmb2>'*self.args.num_query_token+'</s>'+p[3] for p in smiles_prompt]
        # print(smiles_prompt)
        # smiles_prompt = [smiles_handler(p[0], self.mol_ph)[0]+'</s>'+smiles_handler(p[1], self.mol_ph)[0]+'</s>'+p[2]+self.mol_ph+p[3] for p in smiles_prompt]

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

        is_start_token = text_tokens.input_ids == self.tokenizer.start_property_token_id
        text_tokens['is_start_property'] = is_start_token

        is_end_token = text_tokens.input_ids == self.tokenizer.end_property_token_id
        text_tokens['is_end_property'] = is_end_token
        
        
        return graph_batch1,graph_batch2,smiles_prompt_tokens,text_tokens 
          
    
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
        dictionary=None,
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
        
        self.dictionary = dictionary

        self.zijiegou = zijiegou
        

        self.pretrain_dataset =  MoleculeCaption_double(root, text_max_len, self.dictionary,self.prompt,False,True,self.args.solve)
        self.train_dataset =  MoleculeCaption_double(root, text_max_len,self.dictionary, self.prompt,False,True,self.args.solve)
        # self.val_dataset =  MoleculeCaption_double('qformer_data/val/', text_max_len, self.prompt)
        self.val_dataset =  MoleculeCaption_double(args.valid_root, text_max_len, self.dictionary,self.prompt,False,True,self.args.solve)
        self.test_dataset = MoleculeCaption_double(args.valid_root, text_max_len, self.dictionary,self.prompt,False,True,self.args.solve)
        
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
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,args=self.args),
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
                collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.dictionary.pad(),self.zijiegou,args=self.args),
            )
        else:
            raise NotImplementedError
        return loader

    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.dictionary.pad(),self.zijiegou,args=self.args),
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
            collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id,self.zijiegou,args=self.args),
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
