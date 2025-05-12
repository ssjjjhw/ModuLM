import torch
from torch_geometric.data import Dataset
import os
import pandas as pd
import numpy as np


import re
from rdkit import Chem
from rdkit.Chem import BRICS

def extract_smiles(input_string: str) -> str:
    """
    从给定的字符串中提取 SMILES 表达式。
    
    参数:
        input_string (str): 包含 SMILES 表达式的字符串。
        
    返回:
        str: 提取的 SMILES 表达式。如果未找到，则返回 None。
    # """
    # 使用正则表达式提取 SMILES
    match = re.search(r'\[START_I_SMILES\](.*?)\[END_I_SMILES\]', input_string)
    return match.group(1)

import numpy as np
import pickle
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Input, MaxPooling1D, Dropout, Activation
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from pathlib import Path
import click
from rdkit import Chem
from rdkit import RDLogger
from scipy.interpolate import interp1d


#####构象
import numpy as np
import torch
from scipy.spatial import distance_matrix


def process_molecule_data(smi,
    atoms, coordinates_list, dictionary, 
    max_atoms=512, remove_hydrogen=True, 
    remove_polar_hydrogen=False, normalize_coords=True, add_special_token=True
):
    """
    处理分子数据，生成特征张量，支持处理多个构象。

    Args:
        smi (str): SMILES 数据。
        atoms (list[str]): 原子序列。
        coordinates_list (list[np.ndarray]): 多个构象坐标列表，每个元素形状为 [num_atoms, 3]。
        dictionary (object): 用于将原子转为向量的字典对象。
        max_atoms (int): 最大原子数量，超过将被裁剪。
        remove_hydrogen (bool): 是否移除氢原子。
        remove_polar_hydrogen (bool): 是否移除极性氢。
        normalize_coords (bool): 是否归一化坐标。
        add_special_token (bool): 是否添加特殊 token。

    Returns:
        List[Tuple[torch.Tensor]]: 每个构象处理结果的列表，每个结果包括：
            - atom_vec (torch.Tensor): 原子向量表示。
            - coordinates (torch.Tensor): 坐标张量，形状 [num_atoms, 3]。
            - edge_type (torch.Tensor): 边类型张量。
            - dist (torch.Tensor): 距离矩阵张量，形状 [num_atoms, num_atoms]。
            - smi (str): 输入的 SMILES 表示。
    """
    num_types = len(dictionary)
    bos = dictionary.bos()
    eos = dictionary.eos()
    atoms = np.array(atoms)  # 转为 NumPy 数组

    results = []  # 存储所有构象的处理结果

    for coordinates in coordinates_list:
        coordinates = coordinates.astype(np.float32)  # [num_atoms, 3]

        # print(len(atoms),len(coordinates))
        assert len(atoms) == len(coordinates) and len(atoms) > 0  # 确保原子数量和坐标匹配
        assert coordinates.shape[1] == 3  # 坐标必须为 3 维

        # 移除氢原子
        if remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms_filtered = atoms[mask_hydrogen]
                coordinates_filtered = coordinates[mask_hydrogen]
            else:
                atoms_filtered, coordinates_filtered = atoms, coordinates
        else:
            atoms_filtered, coordinates_filtered = atoms, coordinates

        # 移除极性氢
        if not remove_hydrogen and remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms_filtered[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms_filtered = atoms_filtered[:-end_idx]
                coordinates_filtered = coordinates_filtered[:-end_idx]

        # 裁剪到最大原子数量
        if len(atoms_filtered) > max_atoms:
            index = np.random.permutation(len(atoms_filtered))[:max_atoms]
            atoms_filtered = atoms_filtered[index]
            coordinates_filtered = coordinates_filtered[index]

        assert 0 < len(atoms_filtered) <= max_atoms

        # 转换为原子向量
        atom_vec = torch.from_numpy(dictionary.vec_index(atoms_filtered)).long()

        # 坐标归一化
        if normalize_coords:
            coordinates_filtered = coordinates_filtered - coordinates_filtered.mean(axis=0)

        # 添加特殊 token
        if add_special_token:
            atom_vec = torch.cat([torch.LongTensor([bos]), atom_vec, torch.LongTensor([eos])])
            coordinates_filtered = np.concatenate([np.zeros((1, 3)), coordinates_filtered, np.zeros((1, 3))], axis=0)

        # 边类型计算
        edge_type = atom_vec.view(-1, 1) * num_types + atom_vec.view(1, -1)

        # 距离矩阵计算
        dist = distance_matrix(coordinates_filtered, coordinates_filtered).astype(np.float32)

        # 转为 Torch 张量
        coordinates_filtered = torch.from_numpy(coordinates_filtered)  # [num_atoms, 3]
        dist = torch.from_numpy(dist)  # [num_atoms, num_atoms]

        # 将结果添加到列表
        results.append((atom_vec, coordinates_filtered, edge_type, dist,smi))

    return results


######

functional_groups = {
    'Acid anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])'),
    'Acyl halide': Chem.MolFromSmarts('[CX3](=[OX1])[F,Cl,Br,I]'),
    'Alcohol': Chem.MolFromSmarts('[#6][OX2H]'),
    'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)[#6,H]'),
    'Alkane': Chem.MolFromSmarts('[CX4;H3,H2]'),
    'Alkene': Chem.MolFromSmarts('[CX3]=[CX3]'),
    'Alkyne': Chem.MolFromSmarts('[CX2]#[CX2]'),
    'Amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6]'),
    'Amine': Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]'),
    'Arene': Chem.MolFromSmarts('[cX3]1[cX3][cX3][cX3][cX3][cX3]1'),
    'Azo compound': Chem.MolFromSmarts('[#6][NX2]=[NX2][#6]'),
    'Carbamate': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[OX2H0]'),
    'Carboxylic acid': Chem.MolFromSmarts('[CX3](=O)[OX2H]'),
    'Enamine': Chem.MolFromSmarts('[NX3][CX3]=[CX3]'),
    'Enol': Chem.MolFromSmarts('[OX2H][#6X3]=[#6]'),
    'Ester': Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]'),
    'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
    'Haloalkane': Chem.MolFromSmarts('[#6][F,Cl,Br,I]'),
    'Hydrazine': Chem.MolFromSmarts('[NX3][NX3]'),
    'Hydrazone': Chem.MolFromSmarts('[NX3][NX2]=[#6]'),
    'Imide': Chem.MolFromSmarts('[CX3](=[OX1])[NX3][CX3](=[OX1])'),
    'Imine': Chem.MolFromSmarts('[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]'),
    'Isocyanate': Chem.MolFromSmarts('[NX2]=[C]=[O]'),
    'Isothiocyanate': Chem.MolFromSmarts('[NX2]=[C]=[S]'),
    'Ketone': Chem.MolFromSmarts('[#6][CX3](=O)[#6]'),
    'Nitrile': Chem.MolFromSmarts('[NX1]#[CX2]'),
    'Phenol': Chem.MolFromSmarts('[OX2H][cX3]:[c]'),
    'Phosphine': Chem.MolFromSmarts('[PX3]'),
    'Sulfide': Chem.MolFromSmarts('[#16X2H0]'),
    'Sulfonamide': Chem.MolFromSmarts('[#16X4]([NX3])(=[OX1])(=[OX1])[#6]'),
    'Sulfonate': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]'),
    'Sulfone': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[#6]'),
    'Sulfonic acid': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[OX2H]'),
    'Sulfoxide': Chem.MolFromSmarts('[#16X3]=[OX1]'),
    'Thial': Chem.MolFromSmarts('[CX3H1](=S)[#6,H]'),
    'Thioamide': Chem.MolFromSmarts('[NX3][CX3]=[SX1]'),
    'Thiol': Chem.MolFromSmarts('[#16X2H]')
}

def match_group(mol: Chem.Mol, func_group) -> list:
    """返回匹配的官能团子结构列表（包含字典格式的SMILES和原子索引）。"""
    matches = mol.GetSubstructMatches(func_group)
    matched_info = []
    
    for match in matches:
        # 将每个匹配的子结构转为SMILES字符串
        substructure_smiles = Chem.MolFragmentToSmiles(mol, match)
        matched_info.append({
            'fragment': substructure_smiles,
            'atom_indices': list(match)  # 将元组转换为列表以便于返回
        })
    
    return matched_info if matched_info else []

def get_functional_groups(smiles: str) -> list:
    """返回匹配的官能团子结构列表（包含字典格式的SMILES和原子索引）。"""
    RDLogger.DisableLog('rdApp.*')
    smiles = smiles.strip().replace(' ', '')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None  # 无效的SMILES返回None

    matched_substructures = []
    # 遍历功能基团字典，检查分子中是否包含每个基团
    for func_group_name, smarts in functional_groups.items():
        matched_structures = match_group(mol, smarts)
        if matched_structures:  # 如果匹配成功
            # 将所有匹配的子结构和对应的原子索引添加到列表中
            matched_substructures.extend(matched_structures)
    
    # 如果没有匹配到任何子结构，返回原SMILES和空的原子索引
    if len(matched_substructures) == 0:
        matched_substructures.append({
            'fragment': smiles,
            'atom_indices': []
        })
    
    return matched_substructures

def count_subdirectories(folder_path):
    try:
        # 获取文件夹下的所有文件和子文件夹名
        entries = os.listdir(folder_path)

        # 过滤出子文件夹
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        # 返回子文件夹的数量
        return len(subdirectories)
    except FileNotFoundError:
        print(f"文件夹 '{folder_path}' 不存在。")
        return -1  # 返回 -1 表示文件夹不存在
    except Exception as e:
        print(f"发生错误：{e}")
        return -2  # 返回 -2 表示发生了其他错误
class MoleculeCaption(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list) + '\n'

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return data_graph, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token

class MoleculeCaption_double(Dataset):
    def __init__(self, root, text_max_len, dictionary=None,prompt=None,zijiegou=False,autozijiegou=False,solve=False):
        super(MoleculeCaption_double, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        self.dictionary = dictionary

        self.zijiegou = zijiegou
        self.autozijiegou = autozijiegou
        self.solve = solve
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        #atoms coords
        data1 = np.load(graph_path)

        #list list[0] "atom_vec, coordinates_filtered, edge_type, dist"
        

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt

        if len(data1['atoms'])==0:
            return self.__getitem__(index + 1)  # Recursive call to load next item

        data_graph1 = process_molecule_data(smiles,data1['atoms'],data1['coords'],self.dictionary)
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        # data_graph2 = torch.load(graph_path)
        data2 = np.load(graph_path)
        

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
        
        if  len(data2['atoms'])==0:
            return self.__getitem__(index + 1)  # Recursive call to load next item

        data_graph2 = process_molecule_data(smiles,data2['atoms'],data2['coords'],self.dictionary)
        

        if self.solve == False:
            smiles_prompt1 = '</s>'+'The first molecule is'+smiles_prompt1
            smiles_prompt2 = ' </s>'+'.The second molecule is'+smiles_prompt2
            smiles_prompt3 = '.The relationship between them is'#后续处理添加moltokenid作为子结构交互作用的嵌入
            smiles_prompt4 = '.Do they have side effect?'
            smiles_prompt = [smiles_prompt1, smiles_prompt2, smiles_prompt3, smiles_prompt4]
        else:
            smiles_prompt1 = '</s>'+'The first molecule is'+smiles_prompt1
            smiles_prompt2 = ' </s>'+'.The second molecule is'+smiles_prompt2
            smiles_prompt3 = '.The relationship between them is'#后续处理添加moltokenid作为子结构交互作用的嵌入
            smiles_prompt4 = '.What is the solvation Gibbs free energy of this pair of molecules?'
            smiles_prompt = [smiles_prompt1, smiles_prompt2, smiles_prompt3, smiles_prompt4]
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)


        return data_graph1,data_graph2, text ,smiles_prompt

    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
    
class MoleculeCaption_double_old(Dataset):
    def __init__(self, root, text_max_len, prompt=None,zijiegou=False):
        super(MoleculeCaption_double, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        self.zijiegou = zijiegou
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        # smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s> '+smiles_prompt2+' </s>.'
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s> '+smiles_prompt2+' </s>.'+'Answer:'
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        # print(smiles_prompt1,'*********',smiles_prompt2)
        # print(extract_smiles(smiles_prompt1),'sssssssss',extract_smiles(smiles_prompt2))
        
        if self.zijiegou==True:
            zijiegou1 = get_functional_groups(extract_smiles(smiles_prompt1))
            zijiegou2 = get_functional_groups(extract_smiles(smiles_prompt2))
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        if self.zijiegou==False:
            return data_graph1,data_graph2, text ,smiles_prompt
        else:
            return data_graph1,data_graph2, text ,smiles_prompt,zijiegou1,zijiegou2
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token

class MoleculeCaption_double_value(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_value, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
            #return 100
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+" What is the solvation Gibbs free energy of this pair of molecules?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token    
    
class MoleculeCaption_double_DDIvalue(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_DDIvalue, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+" What are the side effects of these two drugs?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token    
class MoleculeCaption_double_fgtvalue(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_fgtvalue, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
            #return 100
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            

        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+" What is the Emission max?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
class MoleculeCaption_universal(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_universal, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        query_name_list = os.listdir(self.root+'query/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt

        query_path = os.path.join(self.root, 'query/'+str(index)+'/', query_name_list[0])
        with open(query_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            query = lines[0].strip()
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> . '+query
        #smiles_prompt = smiles_prompt1+"The front is the first molecule, followed by the second molecule."+smiles_prompt2+"What are the side effects of these two drugs?"
        # load and process text

        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token  
    
if __name__ == '__main__':
    import numpy as np
    pretrain = MoleculeCaption('../data/PubChemDataset_v4/pretrain/', 1000, '')
    train = MoleculeCaption('../data/PubChemDataset_v4/train/', 1000, '')
    valid = MoleculeCaption('../data/PubChemDataset_v4/valid/', 1000, '')
    test = MoleculeCaption('../data/PubChemDataset_v4/test/', 1000, '')

    for subset in [pretrain, train, valid, test]:
        g_lens = []
        t_lens = []
        for i in range(len(subset)):  
            data_graph, text, _ = subset[i]
            g_lens.append(len(data_graph.x))
            t_lens.append(len(text.split()))
            # print(len(data_graph.x))
        g_lens = np.asarray(g_lens)
        t_lens = np.asarray(t_lens)
        print('------------------------')
        print(g_lens.mean())
        print(g_lens.min())
        print(g_lens.max())
        print(t_lens.mean())
        print(t_lens.min())
        print(t_lens.max())
