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



######
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
import os
import numpy as np
import torch
import random
from functools import lru_cache
# from unicore.data import data_utils
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
import lmdb
from rdkit import Chem
import re
import pickle
import multiprocessing
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')




def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
    '''
    This function is responsible for converting a SMILES (Simplified Molecular Input Line Entry System) string into 3D coordinates for each atom in the molecule. It also allows for the generation of 2D coordinates if 3D conformation generation fails, and optionally removes hydrogen atoms and their coordinates from the resulting data.

    :param smi: (str) The SMILES representation of the molecule.
    :param seed: (int, optional) The random seed for conformation generation. Defaults to 42.
    :param mode: (str, optional) The mode of conformation generation, 'fast' for quick generation, 'heavy' for more attempts. Defaults to 'fast'.
    :param remove_hs: (bool, optional) Whether to remove hydrogen atoms from the final coordinates. Defaults to True.

    :return: A tuple containing the list of atom symbols and their corresponding 3D coordinates.
    :raises AssertionError: If no atoms are present in the molecule or if the coordinates do not align with the atom count.
    '''
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms) > 0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms), 3))
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates


def inner_smi2multicoords(smi):
    """
    将 SMILES 表示的分子转换为其原子对应的 3D 坐标，同时移除氢原子。

    Args:
        smi (str): 分子的 SMILES 表示。

    Returns:
        Tuple:
            - atoms (list[str]): 不包含氢原子的原子符号列表。
            - conformer_coords (list[np.ndarray]): 对应构象的 3D 坐标列表，每个元素形状为 [num_atoms, 3]。

    Raises:
        AssertionError: 如果分子中没有原子，或者坐标与原子数量不一致。
    """
    num_conformers = 20
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms) > 0, f"No atoms in molecule: {smi}"
    conformer_coords = []

    try:
        if len(smi) > 500:
            # For large SMILES, generate 2D coordinates
            mols = [Chem.Mol(mol) for _ in range(num_conformers)]
            for mol in mols:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                assert len(atoms) == len(coordinates_2d), f"Coordinates shape does not align with {smi}"
                # Remove hydrogen atoms
                mask = np.array(atoms) != "H"
                conformer_coords.append(coordinates_2d[mask])
        else:
            # Generate 3D conformers
            params = AllChem.ETKDG()
            params.numThreads = 20
            status_list = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
            for i, status in enumerate(status_list):
                conf = mol.GetConformer(i)
                coords = conf.GetPositions().astype(np.float32)
                assert len(atoms) == len(coords), f"Coordinates shape does not align with {smi}"
                # Remove hydrogen atoms
                mask = np.array(atoms) != "H"
                conformer_coords.append(coords[mask])
    except Exception as e:
        print(f"Failed to generate conformer for {smi}, replace with zeros. Error: {e}")
        mask = np.array(atoms) != "H"
        conformer_coords = [np.zeros((sum(mask), 3))] * num_conformers

    while len(conformer_coords) < num_conformers:
        conformer_coords.append(np.zeros((len(atoms), 3)))

    # Filter out hydrogen atoms from the atom list
    atoms = [atom for atom in atoms if atom != "H"]

    return atoms, conformer_coords

######



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
    rdkit_indices = []  # 用于存储 RDKit 原子索引
    idx = 0
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
        rdkit_indices.append(idx)
        idx+=1
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    rdkit_indices = torch.tensor(rdkit_indices, dtype=torch.long)

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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,rdkit_indices=rdkit_indices)
    # data.rdkit_indices = rdkit_indices
    
    return data

def extract_texts_from_csv(csv_path, id1, id2):
    text1, text2 = None, None

    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',', 1)  # Split only at the first comma
            if len(parts) == 2:
                current_id, text = parts
                if current_id == id1:
                    text1 = text
                elif current_id == id2:
                    text2 = text
            if text1 is not None and text2 is not None:
                break  # Stop searching if both texts are found

    return text1, text2


import pandas as pd
import numpy as np
import os 
import numpy as np
import os
from shutil import copy
import pandas as pd

data = pd.read_csv("data/ddi_data/ZhangDDI_train.csv")
data = np.array(data)
idx = np.random.permutation(len(data))
# idx = np.arange(len(data))
train_idx=idx

for i in range(len(train_idx)):
    # if i==1:
    #     break
    data[train_idx[i]]
    os.makedirs("data/ddi_data/Zhangddi_data_3d/train/smiles1/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/train/smiles2/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/train/graph1/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/train/graph2/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/train/text/"+str(i))

    smiles1 = data[train_idx[i]][2]
    smiles2 = data[train_idx[i]][3]

    atoms1,coords1 = inner_smi2multicoords(smiles1)
    np.savez("data/ddi_data/Zhangddi_data_3d/train/graph1/"+str(i)+'/atoms_coords.npz', atoms=atoms1, coords=coords1)

    atoms2,coords2 = inner_smi2multicoords(smiles2)
    np.savez("data/ddi_data/Zhangddi_data_3d/train/graph2/"+str(i)+'/atoms_coords.npz', atoms=atoms2, coords=coords2)

    
    smiles1 = data[train_idx[i]][2]
    smiles2 = data[train_idx[i]][3]
    file = open("data/ddi_data/Zhangddi_data_3d/train/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/ddi_data/Zhangddi_data_3d/train/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
    
    label = data[train_idx[i]][6]
    if label == 0.0:
        text = "Considering the conformer information of molecule1 and the conformer information of molecule2, there is no side effect between molecule a and molecule b."
    elif label == 1.0:
        text = "Considering the conformer information of molecule1 and the conformer information of molecule2, there is a side effect between molecule a and molecule b."
    text3 = text+'\n'
    file = open("data/ddi_data/Zhangddi_data_3d/train/text/"+str(i)+"/text.txt","w")
    file.write(text3)
    file.close()
    print(i)
data = pd.read_csv("data/ddi_data/ZhangDDI_valid.csv")
data = np.array(data)
idx = np.random.permutation(len(data))
# idx = np.arange(len(data))
valid_idx=idx

for i in range(len(valid_idx)):
    # if i==1:
    #     break
    data[valid_idx[i]]
    os.makedirs("data/ddi_data/Zhangddi_data_3d/valid/smiles1/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/valid/smiles2/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/valid/graph1/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/valid/graph2/"+str(i))
    os.makedirs("data/ddi_data/Zhangddi_data_3d/valid/text/"+str(i))

    smiles1 = data[valid_idx[i]][2]
    smiles2 = data[valid_idx[i]][3]

    # data1 = mol_to_graph_data_obj_simple(data[valid_idx[i]][2])
    atoms1,coords1 = inner_smi2multicoords(smiles1)
    np.savez("data/ddi_data/Zhangddi_data_3d/valid/graph1/"+str(i)+'/atoms_coords.npz', atoms=atoms1, coords=coords1)

    atoms2,coords2 = inner_smi2multicoords(smiles2)
    np.savez("data/ddi_data/Zhangddi_data_3d/valid/graph2/"+str(i)+'/atoms_coords.npz', atoms=atoms2, coords=coords2)
    

    file = open("data/ddi_data/Zhangddi_data_3d/valid/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/ddi_data/Zhangddi_data_3d/valid/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
    
    label = data[valid_idx[i]][6]
    if label == 0.0:
        text = "Considering the conformer information of molecule1 and the conformer information of molecule2, there is no side effect between molecule a and molecule b."
    elif label == 1.0:
        text = "Considering the conformer information of molecule1 and the conformer information of molecule2, there is a side effect between molecule a and molecule b."

    text3 = text+'\n'
    file = open("data/ddi_data/Zhangddi_data_3d/valid/text/"+str(i)+"/text.txt","w")
    file.write(text3)
    file.close()
    print(i)
