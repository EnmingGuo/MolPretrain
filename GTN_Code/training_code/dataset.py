import csv
import math
import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)

    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index]) # convert to Chem

        N = mol.GetNumAtoms() # atom number
        M = mol.GetNumBonds() # edge number

        data_i = HeteroData()
        data_j = HeteroData()

        type_idx = []
        chirality_idx = []
        atomic_number = []

        # operate in nodes
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum())) # atom number
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag())) # chirality
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        # self-supervised mask operation
        num_mask_nodes = max([1, math.floor(0.25*N)]) # compute the number needed for mask
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes) # the serial number of mask nodes
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)

        x_i = deepcopy(x) # deep copy
        for atom_idx in mask_nodes_i: # if not in the mask number, keep the original information
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0]) # if in, replace to [118,0]
        data_i['atom'].x = x_i # node feature

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        data_j['atom'].x = x_j

        # operate in edges
        bond_types = {
            'SINGLE': {'row': [], 'col': [], 'features': []}, # single
            'DOUBLE': {'row': [], 'col': [], 'features': []}, # double
            'TRIPLE': {'row': [], 'col': [], 'features': []}, # triple
            'AROMATIC': {'row': [], 'col': [], 'features': []} # aromatic
        }

        def add_bond(bond_type, start, end, features):
            bond_types[bond_type]['row'] += [start, end] # store edge information
            bond_types[bond_type]['col'] += [end, start]
            bond_types[bond_type]['features'].append(features) # append twice due to undirected graph
            bond_types[bond_type]['features'].append(features)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            features = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            bond_type = str(bond.GetBondType())
            if bond_type in bond_types:
                add_bond(bond_type, start, end, features)
            else:
                print('unknown edge type!')

        def process_bond(bond_type):
            """create masked edge information"""
            index = torch.tensor([bond_types[bond_type]['row'], bond_types[bond_type]['col']], dtype=torch.float32)
            num_bond = int(index.shape[-1] / 2)
            num_mask_edges = max([0, math.floor(0.25 * num_bond)])
            mask_edges_i = random.sample(list(range(num_bond)), num_mask_edges) # masked edge number
            mask_edges_j = random.sample(list(range(num_bond)), num_mask_edges)
            mask_edges_i = [2 * i for i in mask_edges_i] + [2 * i + 1 for i in mask_edges_i]
            mask_edges_j = [2 * i for i in mask_edges_j] + [2 * i + 1 for i in mask_edges_j]
            return mask_edges_i, mask_edges_j

        # mask four kinds of edges
        mask_edges_single_i, mask_edges_single_j = process_bond('SINGLE')
        mask_edges_double_i, mask_edges_double_j = process_bond('DOUBLE')
        mask_edges_triple_i, mask_edges_triple_j = process_bond('TRIPLE')
        mask_edges_aromatic_i, mask_edges_aromatic_j = process_bond('AROMATIC')


        def process_edge(bond_type, mask_edges_i, mask_edges_j):
            """final edge process information"""
            index = torch.tensor([bond_types[bond_type]['row'], bond_types[bond_type]['col']])
            attr = torch.tensor(np.array(bond_types[bond_type]['features']))
            num_bond = int(index.shape[-1] / 2)
            num_mask_edges = max([0, math.floor(0.25 * num_bond)])
            edge_index_i = torch.zeros((2, 2 * (num_bond - num_mask_edges)), dtype=torch.long)
            edge_attr_i = torch.zeros((2 * (num_bond - num_mask_edges), 2), dtype=torch.long)
            count = 0
            for bond_idx in range(2 * num_bond):
                if bond_idx not in mask_edges_i:
                    edge_index_i[:, count] = index[:, bond_idx]
                    edge_attr_i[count, :] = attr[bond_idx, :]
                    count += 1

            edge_index_j = torch.zeros((2, 2 * (num_bond - num_mask_edges)), dtype=torch.long)
            edge_attr_j = torch.zeros((2 * (num_bond - num_mask_edges), 2), dtype=torch.long)
            count = 0
            for bond_idx in range(2 * num_bond):
                if bond_idx not in mask_edges_j:
                    edge_index_j[:, count] = index[:, bond_idx]
                    edge_attr_j[count, :] = attr[bond_idx, :]
                    count += 1

            return edge_index_i, edge_attr_i, edge_index_j, edge_attr_j

        edge_index_i_single, edge_attr_i_single, edge_index_j_single, edge_attr_j_single = process_edge('SINGLE', mask_edges_single_i, mask_edges_single_j)
        edge_index_i_double, edge_attr_i_double, edge_index_j_double, edge_attr_j_double = process_edge('DOUBLE', mask_edges_double_i, mask_edges_double_j)
        edge_index_i_triple, edge_attr_i_triple, edge_index_j_triple, edge_attr_j_triple = process_edge('TRIPLE', mask_edges_triple_i, mask_edges_triple_j)
        edge_index_i_aromatic, edge_attr_i_aromatic, edge_index_j_aromatic, edge_attr_j_aromatic = process_edge('AROMATIC', mask_edges_aromatic_i, mask_edges_aromatic_j)

        # operate in four kind of edges
        data_i['atom', 'SINGLE', 'atom'].edge_index = edge_index_i_single
        data_i['atom', 'DOUBLE', 'atom'].edge_index = edge_index_i_double
        data_i['atom', 'TRIPLE', 'atom'].edge_index = edge_index_i_triple
        data_i['atom', 'AROMATIC', 'atom'].edge_index = edge_index_i_aromatic
        data_i['atom', 'SINGLE', 'atom'].edge_attr = edge_attr_i_single
        data_i['atom', 'DOUBLE', 'atom'].edge_attr = edge_attr_i_double
        data_i['atom', 'TRIPLE', 'atom'].edge_attr = edge_attr_i_triple
        data_i['atom', 'AROMATIC', 'atom'].edge_attr = edge_attr_i_aromatic

        data_j['atom', 'SINGLE', 'atom'].edge_index = edge_index_j_single
        data_j['atom', 'DOUBLE', 'atom'].edge_index = edge_index_j_double
        data_j['atom', 'TRIPLE', 'atom'].edge_index = edge_index_j_triple
        data_j['atom', 'AROMATIC', 'atom'].edge_index = edge_index_j_aromatic
        data_j['atom', 'SINGLE', 'atom'].edge_attr = edge_attr_j_single
        data_j['atom', 'DOUBLE', 'atom'].edge_attr = edge_attr_j_double
        data_j['atom', 'TRIPLE', 'atom'].edge_attr = edge_attr_j_triple
        data_j['atom', 'AROMATIC', 'atom'].edge_attr = edge_attr_j_aromatic
        
        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)
    
    def get(self):
        pass

    def len(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # single GPU: DataLoader multi GPU:DataListLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
