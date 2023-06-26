import csv
import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict
import pandas as pd
from argparse import Namespace
from typing import List, Tuple, Union


def construct_seq_index():
    mole_dict = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: " Ne",
                11: "Na", 12:"Mg", 13: "Al", 14:"Si", 15:"P", 16: "S", 17: "Cl", 18:"Ar", 19:"K", 20:"Ca", 22:"Ti", 24:"Cr", 26:"Fe", 28:"Ni",
                29:"Cu", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 40:"Zr", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 50:"Sn", 51:"Sb", 52:"Te", 53: "I", 65:"Tb", 75:"Re", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
                81:"Tl", 82:"Pb", 83:"Bi"}

    pair_list = ["Br", "Cl", "Si", "Na", "Ca", "Ge", "Cu", "Au", "Sn", "Tb", "Pt", "Re", "Ru", "Bi", "Li", "Fe", "Sb", "Hg","Pb", "Se" ,'se',"Ag","Cr","Pd","Ga","Mg","Ni","Ir","Rh","Te","Ti","Al","Zr","Tl","As"]
    additional_list = ['s', '@', 'N', 'H', ']', '[', 'C', 'B', 'n', '}', '<', '=', '0', '8', 'o', ',', '>', '2', '9', '%', '#', '(', '+',
     'O', '-', '4', '.', ')', 'l', '5', 'r', '7', 'S', '{', '3', 'c', '6', '1', 'F']

    #bond_dict = {'SINGLE': 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    node_types = set()
    seq_types = set()

    node_types|= set(additional_list)
    seq_types |= set(additional_list)
    seq_types=additional_list
    # print(seq_types)
    # print('max_seq_count:'+ str(max_seq_count))
    node2index = {n: i for i, n in enumerate(node_types)}
    seq2index = {n: i for i, n in enumerate(seq_types)}
    # return node2index, seq2index,max_seq_count,list(seq_types)
    return seq2index



def get_smiles_feature(smile_list,index):
    smile_features = []
    smile_sequence = []
    for smile in smile_list:
        feature = torch.zeros(len(smile), len(index)).long()
        se_num = 0
        i = 0
        smiles_seq = []
        while i < len(smile):
            seq_str = smile[i]
            # print(seq_str)
            # this_str = smile[i]
            # if i < len(smile)-1 and (smile[i] + smile[i+1]) in pair_list:
            #     if (smile[i] + smile[i+1]) !='se': ###
            #         this_str = smile[i] + smile[i + 1]
            #         seq_str = smile[i] + smile[i+1]
            #         i += 2
            #     else:  ###
            #         this_str = smile[i].upper() + smile[i + 1]
            #         seq_str = smile[i] + smile[i + 1]
            #         i += 2
            # else:
            #     seq_str = seq_str
            #     this_str = this_str.upper()
            #     i += 1
            smiles_seq.append(index[seq_str])
            # if this_str in graph_nodes and this_str == mole_dict[mol.GetAtoms()[gr_num].GetAtomicNum()]:
            #     map[gr_num] = se_num
            #     #print(map[gr_num])
            #     gr_num += 1
            feature[se_num, index[seq_str]] = 1
            i = i + 1
            se_num += 1
            # print(smiles_seq)
            # print(map)
            # adj_list = defaultdict(list)

        smile_features.append(torch.LongTensor(feature))
        # smile_sequence.append(torch.tensor(smiles_seq))
        smile_sequence.append(torch.LongTensor(smiles_seq))
    return smile_features,smile_sequence



class SmilesSquence:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles

class BatchSmilesSquence:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, smiles_list: List[SmilesSquence], args: Namespace):
        self.smiles_batch = [smile_list.smiles for smile_list in smiles_list]

    def get_components(self) -> List[str]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.smiles_batch



def smile2smile(smiles_batch: List[str],
              args: Namespace) -> BatchSmilesSquence:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    smiles_list = []
    for smiles in smiles_batch:
        smile_list = SmilesSquence(smiles, args)
        smiles_list.append(smile_list)

    return BatchSmilesSquence(smiles_list, args)