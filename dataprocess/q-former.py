import os
import re
import torch
import random
import pickle
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain
# import transformers
def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges

def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(smiles):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def extract_texts_from_csv(csv_path, id1, id2):
    text1, text2 = None, None

    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',', 1)  
            if len(parts) == 2:
                current_id, text = parts
                if current_id == id1:
                    text1 = text
                elif current_id == id2:
                    text2 = text
            if text1 is not None and text2 is not None:
                break  

    return text1, text2


import pandas as pd
import numpy as np
import os 
import numpy as np
import os
from shutil import copy
import pandas as pd
data = pd.read_csv("data/ddi_data/drug2.csv")
#text = pd.read_csv('drug.csv')
#首先解决testing的数据集
#data = pd.read_csv('drug2.csv')
#text = np.array(text)
data = np.array(data)
def calculate_bounds(number):
    lower_bound = math.floor(number * 2) / 2  # 下界
    upper_bound = math.ceil(number * 2) / 2   # 上界

    return lower_bound, upper_bound
#data = np.array(data)
#print(len(data))
#data/ddi_data/train/
idx = np.random.permutation(len(data))
#train_idx=idx
train_idx = idx[0:99000]
valid_idx = idx[99000:100000]
for i in range(len(train_idx)):
    # if i==10:
    #     break
    data[train_idx[i]]
    os.makedirs("qformer_data/train/smiles1/"+str(i))
    os.makedirs("qformer_data/train/smiles2/"+str(i))
    os.makedirs("qformer_data/train/graph1/"+str(i))
    os.makedirs("qformer_data/train/graph2/"+str(i))
    os.makedirs("qformer_data/train/text/"+str(i))


    data1 = mol_to_graph_data_obj_simple(data[train_idx[i]][2])
    torch.save(data1,"qformer_data/train/graph1/"+str(i)+'/graph_data.pt')
    data1 = mol_to_graph_data_obj_simple(data[train_idx[i]][3])
    torch.save(data1,"qformer_data/train/graph2/"+str(i)+'/graph_data.pt')
    #value = data[train_idx[i]][4]
    #lower, upper = calculate_bounds(value)
    #text = "The solvation Gibbs free energy of these two molecules is above "+str(lower)+" and below "+str(upper)+", so the accurate value is "+str('%.2f'%value)+";"
    #file = open("ddi_data/train/text/"+str(i)+"/text.txt","w")
    #file.write(text)
    #file.close()
    smiles1 = data[train_idx[i]][2]
    smiles2 = data[train_idx[i]][3]
    file = open("qformer_data/train/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("qformer_data/train/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
    id1=data[train_idx[i]][0]
    id2=data[train_idx[i]][1]
    csv_path='data/ddi_data/drug.csv'
    text1, text2 = extract_texts_from_csv(csv_path, id1, id2)
    #text = dic["DDI type "+str(data[train_idx[i]][4]+1)
    #text3 = text1+';'+text2+';'+
    #text3 = text+'\n'
    text3 = '</s> '+text1+' </s>'+' </s> '+text2+' </s> '+'\n'
    file = open("qformer_data/train/text/"+str(i)+"/text.txt","w")
    file.write(text3)
    file.close()
    print(i)
for i in range(len(valid_idx)):
    # if i==10:
    #     break
    #data[train_idx[i]]
    os.makedirs("qformer_data/val/smiles1/"+str(i))
    os.makedirs("qformer_data/val/smiles2/"+str(i))
    os.makedirs("qformer_data/val/graph1/"+str(i))
    os.makedirs("qformer_data/val/graph2/"+str(i))
    os.makedirs("qformer_data/val/text/"+str(i))


    data1 = mol_to_graph_data_obj_simple(data[valid_idx[i]][2])
    torch.save(data1,"qformer_data/val/graph1/"+str(i)+'/graph_data.pt')
    data1 = mol_to_graph_data_obj_simple(data[valid_idx[i]][3])
    torch.save(data1,"qformer_data/val/graph2/"+str(i)+'/graph_data.pt')
    #value = data[valid_idx[i]][4]
    #lower, upper = calculate_bounds(value)
    #text = "The solvation Gibbs free energy of these two molecules is above "+str(lower)+" and below "+str(upper)+", so the accurate value is "+str('%.2f'%value)+";"
    #file = open("ddi_data4/val/text/"+str(i)+"/text.txt","w")
    #file.write(text)
    #file.close()
    smiles1 = data[valid_idx[i]][2]
    smiles2 = data[valid_idx[i]][3]
    file = open("qformer_data/val/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("qformer_data/val/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
    id1=data[valid_idx[i]][0]
    id2=data[valid_idx[i]][1]
    csv_path='data/ddi_data/drug.csv'
    text1, text2 = extract_texts_from_csv(csv_path, id1, id2)
    #text = dic["DDI type "+str(data[i][4]+1)]
    #text3 = text1+';'+text2+';'+text
    #text3 = text+'\n'
    text3 = '</s> '+text1+' </s>'+' </s> '+text2+' </s> '+'\n'
    file = open("qformer_data/val/text/"+str(i)+"/text.txt","w")
    file.write(text3)
    file.close()
    print(i) 
