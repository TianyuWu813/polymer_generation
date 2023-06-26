import numpy as np
import pandas as pd
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset

from .utils import Variable

import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
import scaffold.utils.chem as uc
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

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
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


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['EOS', 'GO']
        # self.special_tokens = ['EOS', 'GO', 'is_DRD2', 'not_DRD2',  'high_QED', 'low_QED',
                               # 'good_SA', 'bad_SA']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    # def tokenize(self, smiles):
    #     """Takes a SMILES and return a list of characters/tokens"""
    #     regex = '(\[[^\[\]]{1,6}\])'
    #     smiles = replace_halogen(smiles)
    #     print(smiles,1234)
    #     char_list = re.split(regex, smiles)
    #     print(char_list, 1234)
    #     tokenized = []
    #     for char in char_list:
    #         if char.startswith('['):
    #             tokenized.append(char)
    #         else:
    #             chars = [unit for unit in char]
    #             [tokenized.append(unit) for unit in chars]
    #     tokenized.append('EOS')
    #     return tokenized

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)

        char_list = re.split(regex, smiles)

        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    # def token_conditional_token(self, con_list):
    #     for i, char in enumerate(con_list):
    #         smiles_matrix[i] = self.vocab[char]


    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """

    def __init__(self, fname, voc):
        self.voc = voc
        # self.smiles = []
        df = pd.read_csv(fname,header=None)
        df.columns=['SMILES']
        self.smiles = df['SMILES'].values.tolist()
        self.scaffold =[]
        self.decorator =[]
        for smile in self.smiles:
            piece=smile.split('\t')
            self.scaffold.append(piece[0])
            self.decorator.append(piece[1])

        # convert conditional to token
        # self.con = df[['drd2', 'qed', 'sa']]
        # self.con = self.condition_convert(self.con).values.tolist()

        # for i in range(len(self.smiles)):
        #     # print(self.smiles[i])
        #     data = mol_to_graph_data_obj_simple(AllChem.MolFromSmiles(self.smiles[i]))
        #     self.data_list.append(data)



    def __getitem__(self, i):
        # con_token = self.con[i]
        sca = self.scaffold[i]
        sca_tokenized = self.voc.tokenize(sca)

        dec = self.decorator[i]
        dec_tokenized = self.voc.tokenize(dec)
        # add token to smiles
        # tokenized = con_token + tokenized
        # encoded
        sca_encoded = self.voc.encode(sca_tokenized)
        dec_encoded = self.voc.encode(dec_tokenized)

        # print(list(zip(Variable(sca_encoded),Variable(dec_encoded))))
        # return Variable(encoded),data_i
        return (Variable(sca_encoded),Variable(dec_encoded))

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        scaffold_max_length = max([seq[0].size(0) for seq in arr])
        scaffold_collated_arr = Variable(torch.zeros(len(arr), scaffold_max_length))
        for i, seq in enumerate(arr):
            scaffold_collated_arr[i, :seq[0].size(0)] = seq[0]

        decorator_max_length = max([seq[1].size(0) for seq in arr])
        decorator_collated_arr = Variable(torch.zeros(len(arr), decorator_max_length))
        for i, seq in enumerate(arr):
            decorator_collated_arr[i, :seq[1].size(0)] = seq[1]

        return scaffold_collated_arr,decorator_collated_arr

    # def collate_fn(cls, arr):
    #     """Function to take a list of encoded sequences and turn them into a batch"""
    #     smiles_arr = []
    #     graph_arr = []
    #     for smiles_seq, graph in arr:
    #         smiles_arr.append(smiles_seq)
    #         graph_arr.append(graph)
    #     # print(arr,222)
    #     # print(smiles_arr,111)
    #     # print(graph_arr,123)
    #     max_length = max([seq.size(0) for seq in smiles_arr])
    #     collated_arr = Variable(torch.zeros(len(smiles_arr), max_length))
    #     for i, seq in enumerate(smiles_arr):
    #         collated_arr[i, :seq.size(0)] = seq
    #     return collated_arr,graph_arr

    def condition_convert(self, con_df):
        # convert to 0, 1

        con_df['drd2'][con_df['drd2'] >= 0.5] = 1
        con_df['drd2'][con_df['drd2'] < 0.5] = 0
        con_df['qed'][con_df['qed'] >= 0.6] = 1
        con_df['qed'][con_df['qed'] < 0.6] = 0
        con_df['sa'][con_df['sa'] <= 4.0] = 1
        con_df['sa'][con_df['sa'] > 4.0] = 0

        # convert to token

        con_df['drd2'][con_df['drd2'] == 1] = 'is_DRD2'
        con_df['drd2'][con_df['drd2'] == 0] = 'not_DRD2'
        con_df['qed'][con_df['qed'] == 1] = 'high_QED'
        con_df['qed'][con_df['qed'] == 0] = 'low_QED'
        con_df['sa'][con_df['sa'] == 1] = 'good_SA'
        con_df['sa'][con_df['sa'] == 0] = 'bad_SA'

        return con_df

class MolData_generate(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """

    def __init__(self, fname, voc):
        self.voc = voc
        # self.smiles = []
        df = pd.read_csv(fname,header=None)
        df.columns=['SMILES']
        self.smiles = df['SMILES'].values.tolist()
        self.scaffold =[]
        # self.decorator =[]
        for smile in self.smiles:
            piece=smile.split('\t')
            self.scaffold.append(piece[0])
            # self.decorator.append(piece[1])

        # convert conditional to token
        # self.con = df[['drd2', 'qed', 'sa']]
        # self.con = self.condition_convert(self.con).values.tolist()

        # for i in range(len(self.smiles)):
        #     # print(self.smiles[i])
        #     data = mol_to_graph_data_obj_simple(AllChem.MolFromSmiles(self.smiles[i]))
        #     self.data_list.append(data)



    def __getitem__(self, i):
        # con_token = self.con[i]
        sca = self.scaffold[i]
        sca_tokenized = self.voc.tokenize(sca)

        # dec = self.decorator[i]
        # dec_tokenized = self.voc.tokenize(dec)
        # add token to smiles
        # tokenized = con_token + tokenized
        # encoded
        sca_encoded = self.voc.encode(sca_tokenized)
        # dec_encoded = self.voc.encode(dec_tokenized)

        # print(list(zip(Variable(sca_encoded),Variable(dec_encoded))))
        # return Variable(encoded),data_i
        return Variable(sca_encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    # @classmethod
    # def collate_fn(cls, arr):
    #     """Function to take a list of encoded sequences and turn them into a batch"""
    #     scaffold_max_length = max([seq[0].size(0) for seq in arr])
    #     scaffold_collated_arr = Variable(torch.zeros(len(arr), scaffold_max_length))
    #     for i, seq in enumerate(arr):
    #         scaffold_collated_arr[i, :seq[0].size(0)] = seq[0]
    #
    #     decorator_max_length = max([seq[1].size(0) for seq in arr])
    #     decorator_collated_arr = Variable(torch.zeros(len(arr), decorator_max_length))
    #     for i, seq in enumerate(arr):
    #         decorator_collated_arr[i, :seq[1].size(0)] = seq[1]
    #
    #     return scaffold_collated_arr,decorator_collated_arr

    @classmethod
    def collate_fn(cls, arr):
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr

    # def collate_fn(cls, arr):
    #     """Function to take a list of encoded sequences and turn them into a batch"""
    #     smiles_arr = []
    #     graph_arr = []
    #     for smiles_seq, graph in arr:
    #         smiles_arr.append(smiles_seq)
    #         graph_arr.append(graph)
    #     # print(arr,222)
    #     # print(smiles_arr,111)
    #     # print(graph_arr,123)
    #     max_length = max([seq.size(0) for seq in smiles_arr])
    #     collated_arr = Variable(torch.zeros(len(smiles_arr), max_length))
    #     for i, seq in enumerate(smiles_arr):
    #         collated_arr[i, :seq.size(0)] = seq
    #     return collated_arr,graph_arr

    def condition_convert(self, con_df):
        # convert to 0, 1

        con_df['drd2'][con_df['drd2'] >= 0.5] = 1
        con_df['drd2'][con_df['drd2'] < 0.5] = 0
        con_df['qed'][con_df['qed'] >= 0.6] = 1
        con_df['qed'][con_df['qed'] < 0.6] = 0
        con_df['sa'][con_df['sa'] <= 4.0] = 1
        con_df['sa'][con_df['sa'] > 4.0] = 0

        # convert to token

        con_df['drd2'][con_df['drd2'] == 1] = 'is_DRD2'
        con_df['drd2'][con_df['drd2'] == 0] = 'not_DRD2'
        con_df['qed'][con_df['qed'] == 1] = 'high_QED'
        con_df['qed'][con_df['qed'] == 0] = 'low_QED'
        con_df['sa'][con_df['sa'] == 1] = 'good_SA'
        con_df['sa'][con_df['sa'] == 0] = 'bad_SA'

        return con_df

class MolData_sample_molecule(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """

    def __init__(self, list_canonicalized_scaffold,list_scaffold_with_num,smiles_list,list_attach_point, voc):
        self.voc = voc
        self.canonicalized_scaffold = list_canonicalized_scaffold
        self.list_randomlized_scaffold_with_num = list_scaffold_with_num

        self.scaffold =smiles_list
        self.attach_point = list_attach_point

        self.smiles = smiles_list

        # for smile in self.smiles:
        #     self.scaffold.append(smile)
            # self.decorator.append(piece[1])

        # convert conditional to token
        # self.con = df[['drd2', 'qed', 'sa']]
        # self.con = self.condition_convert(self.con).values.tolist()

        # for i in range(len(self.smiles)):
        #     # print(self.smiles[i])
        #     data = mol_to_graph_data_obj_simple(AllChem.MolFromSmiles(self.smiles[i]))
        #     self.data_list.append(data)



    def __getitem__(self, i):
        # con_token = self.con[i]
        can_sca =self.canonicalized_scaffold[i]
        ran_sca_num = self.list_randomlized_scaffold_with_num[i]
        sca = self.scaffold[i]
        att_point = self.attach_point[i]
        sca_tokenized = self.voc.tokenize(sca)

        # dec = self.decorator[i]
        # dec_tokenized = self.voc.tokenize(dec)
        # add token to smiles
        # tokenized = con_token + tokenized
        # encoded
        sca_encoded = self.voc.encode(sca_tokenized)
        # dec_encoded = self.voc.encode(dec_tokenized)

        # print(list(zip(Variable(sca_encoded),Variable(dec_encoded))))
        # return Variable(encoded),data_i
        return (Variable(sca_encoded),can_sca,ran_sca_num,att_point)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    # @classmethod
    # def collate_fn(cls, arr):
    #     """Function to take a list of encoded sequences and turn them into a batch"""
    #     scaffold_max_length = max([seq[0].size(0) for seq in arr])
    #     scaffold_collated_arr = Variable(torch.zeros(len(arr), scaffold_max_length))
    #     for i, seq in enumerate(arr):
    #         scaffold_collated_arr[i, :seq[0].size(0)] = seq[0]
    #
    #     decorator_max_length = max([seq[1].size(0) for seq in arr])
    #     decorator_collated_arr = Variable(torch.zeros(len(arr), decorator_max_length))
    #     for i, seq in enumerate(arr):
    #         decorator_collated_arr[i, :seq[1].size(0)] = seq[1]
    #
    #     return scaffold_collated_arr,decorator_collated_arr

    @classmethod
    def collate_fn(cls, arr):
        max_length = max([seq[0].size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq[0].size(0)] = seq[0]
        collated_arr_can=[]
        collated_arr_ran = []
        collated_arr_att=[]
        for i, seq in enumerate(arr):
            collated_arr_can.append(seq[1])
        for i, seq in enumerate(arr):
            collated_arr_ran.append(seq[2])
        for i, seq in enumerate(arr):
            collated_arr_att.append(seq[3])
        return collated_arr, collated_arr_can,collated_arr_ran,collated_arr_att,

    # def collate_fn(cls, arr):
    #     """Function to take a list of encoded sequences and turn them into a batch"""
    #     smiles_arr = []
    #     graph_arr = []
    #     for smiles_seq, graph in arr:
    #         smiles_arr.append(smiles_seq)
    #         graph_arr.append(graph)
    #     # print(arr,222)
    #     # print(smiles_arr,111)
    #     # print(graph_arr,123)
    #     max_length = max([seq.size(0) for seq in smiles_arr])
    #     collated_arr = Variable(torch.zeros(len(smiles_arr), max_length))
    #     for i, seq in enumerate(smiles_arr):
    #         collated_arr[i, :seq.size(0)] = seq
    #     return collated_arr,graph_arr

    def condition_convert(self, con_df):
        # convert to 0, 1

        con_df['drd2'][con_df['drd2'] >= 0.5] = 1
        con_df['drd2'][con_df['drd2'] < 0.5] = 0
        con_df['qed'][con_df['qed'] >= 0.6] = 1
        con_df['qed'][con_df['qed'] < 0.6] = 0
        con_df['sa'][con_df['sa'] <= 4.0] = 1
        con_df['sa'][con_df['sa'] > 4.0] = 0

        # convert to token

        con_df['drd2'][con_df['drd2'] == 1] = 'is_DRD2'
        con_df['drd2'][con_df['drd2'] == 0] = 'not_DRD2'
        con_df['qed'][con_df['qed'] == 1] = 'high_QED'
        con_df['qed'][con_df['qed'] == 0] = 'low_QED'
        con_df['sa'][con_df['sa'] == 1] = 'good_SA'
        con_df['sa'][con_df['sa'] == 0] = 'bad_SA'

        return con_df


class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""

    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)

class Experience_scaffold(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""

    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        # print(self.memory)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if (exp[0],exp[3]) not in smiles :
                    idxs.append(i)
                    smiles.append((exp[0],exp[3]))
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('EOS')
    return tokenized


def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list


def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6, 7, 8, 9, 16, 17, 35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False


def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    return smiles_list_valid


def filter_file_on_chars(smiles_fname, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    with open(smiles_fname, 'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    print(smiles[:10])
    chars = []
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print(chars)
    valid_smiles = filter_on_chars(smiles, chars)
    with open(smiles_fname + "_filtered", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")


def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")


def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open('data/Voc_RE', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars


if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_file(smiles_file)
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(smiles_list)
    write_smiles_to_file(smiles_list, "data/mols_filtered.smi")
