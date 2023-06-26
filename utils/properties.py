#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
from utils import scripts as sascorer
import pickle
import utils.scaffold_utils.scaffold as usc
from rdkit import DataStructs
import rdkit
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
# from chemprop.train import predict
# from chemprop.data import MoleculeDataset
# from chemprop.data.utils import get_data, get_data_from_smiles
# from chemprop.utils import load_args, load_checkpoint, load_scalers

rdBase.DisableLog('rdApp.error')

def reward_for_activity(value):
    return 10-1.305 *np.exp(0.7301*value/2.3)

class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/gsk3/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/jnk3/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class qed_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0
                scores.append(qed)
        return np.float32(scores)

class qed_func_scaffold():

    def __call__(self, scaffold_list, decorator_list):
        scores = []
        for sca, dec in zip(scaffold_list, decorator_list):
            mol = usc.join_joined_attachments(sca, dec)
            if mol is None:
                scores.append(0)
            else:
                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0
                scores.append(qed)
        return np.float32(scores)

class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(sascorer.calculateScore(mol))
        return np.float32(scores)

class logP_func():

    def __call__(self, scaffold_list,decorator_list):
        scores = []
        for sca, dec in zip(scaffold_list, decorator_list):
            mol = usc.join_joined_attachments(sca, dec)
            if mol is None:
                scores.append(0)
            else:
                logp = -Chem.Crippen.MolLogP(mol)
                if (logp >= 1.0) and (logp <= 2.0):
                    scores.append(11.0)
                else:
                    scores.append(1.0)
        return np.float32(scores)

class atom_num_func():

    def __call__(self, scaffold_list,decorator_list):
        scores = []
        for sca, dec in zip(scaffold_list, decorator_list):
            mol = usc.join_joined_attachments(sca, dec)
            if mol is None:
                scores.append(-2)
            else:
                num = len(mol.GetAtoms())
                # scores.append(4*(num/10))
                if (num >= 0) and (num <= 10):
                    scores.append(1)
                elif(num >=10) and (num <= 20):
                    scores.append(2)
                elif(num > 20) and (num <= 30):
                    scores.append(8)
                elif (num > 30) :
                    scores.append(10)
        return np.float32(scores)


class drd2_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = './data/drd2/drd2.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = drd2_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

class drd2_model_sca():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = './data/drd2/drd2.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, scaffold_list,decorator_list):
        fps = []
        mask = []
        for sca,dec in zip(scaffold_list,decorator_list):
            mol = usc.join_joined_attachments(sca, dec)
            mask.append(int(mol is not None))
            fp = drd2_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(100*scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

class Property_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = './data/drd2/drd2.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = drd2_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

class scaffold_syntax_func():
    def __call__(self, scaffold_list,decorator_list):
        scores = []
        for sca,dec in zip(scaffold_list,decorator_list):
            mol = usc.join_joined_attachments(sca, dec)
            if mol is None:
                scores.append(-1)
            else:
                scores.append(2)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""
    if prop_name == 'jnk3':
        return jnk3_model()
    elif prop_name == 'gsk3':
        return gsk3_model()
    elif prop_name == 'qed':
        return qed_func()
    elif prop_name == 'qed_sca':
        return qed_func_scaffold()
    elif prop_name == 'sa':
        return sa_func()
    elif prop_name == 'drd2':
        return drd2_model()
    elif prop_name == 'drd2_sca':
        return drd2_model_sca()
    elif prop_name == 'property':
        return Property_model()
    elif prop_name == 'syntax':
        return scaffold_syntax_func()
    elif prop_name == 'logP':
        return logP_func()
    elif prop_name == 'atom_num':
        return atom_num_func()

    # else:
    #     return chemprop_model(prop_name)


def multi_scoring_functions(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    scoring_sum = props.sum(axis=0)

    return scoring_sum


def multi_scoring_functions_one_hot(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_scaffold(activity,activity_2,activity_3,function_list):
    all = [activity,activity_2,activity_3]
    # print(activity)
    props = np.array([act for act in all])
    # print(props,222)

    props = pd.DataFrame(props).T
    props.columns = function_list
    # print(props,222)

    scoring_sum = condition_convert_sca(props).values.sum(1)
    # print(scoring_sum,333)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum


def condition_convert(con_df):
    # convert to 0, 1
    #
    con_df['drd2'][con_df['drd2'] >= 0.5] = 1
    con_df['drd2'][con_df['drd2'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_sca(con_df):
    # convert to 0, 1
    #
    con_df['HC10'][con_df['HC10'] < 0.5] = 0
    con_df['HC10'][con_df['HC10'] >= 0.5] = 1
    con_df['Saureus'][con_df['Saureus'] < 0.6] = 0
    con_df['Saureus'][con_df['Saureus'] >= 0.6] = 1
    con_df['Ecoli'][con_df['Ecoli'] < 0.6] = 0
    con_df['Ecoli'][con_df['Ecoli'] >=0.6] = 1
    # print(con_df)
    return con_df


def ring_calculate(input):
    input_mol = Chem.MolFromSmiles(input)
    ssr = Chem.GetSymmSSSR(input_mol)
    if len(ssr) > 1:
        return -3
    else:
        return 3

smis=[
'NC1CCCCC1C(=O)',
'NC1CCCC1C(=O)',
'NC1CCCCCC1C(=O)','NC1CCCCCCC1C(=O)','NC1CCCCCCCCCCC1C(=O)','NC(CC)C(CC)C(=O)','NC(CC)C(CC)C(=O)','NC(CCC)C(CCC)C(=O)','NC(CC)(CC)CC(=O)',
'NC(C)(C)C(C)(C)C(=O)','NC(CCCC)CC(=O)']
mols =[]
for smi in smis:
        m = Chem.MolFromSmiles(smi)
        mols.append(m)

def sim_calculate(input):
    input_mol = [Chem.MolFromSmiles(input)]

    fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    # fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]

#     input_fps = [Chem.RDKFingerprint(x) for x in input_mol]
    input_fps = [MACCSkeys.GenMACCSKeys(x) for x in input_mol]
    sm_one=[]
    for m in range(len(fps)):
#         sm=DataStructs.FingerprintSimilarity(input_fps[0],fps[m],metric=DataStructs.DiceSimilarity)  #MACCS
        sm=DataStructs.FingerprintSimilarity((input_fps[0]),fps[m])  #RDKIT
        sm_one.append(sm)

    #         sm=DataStructs.BulkTanimotoSimilarity(fps[i],[fps[m]]) #Morgan
    #         sm_one.append(sm[0])
#         print(sm)
    return sum(sm_one)/len(fps)

# def atom_num_calculate(input):
#     targetmol = Chem.MolFromSmiles(input)
#     atoms = targetmol.GetAtoms()
#     ssr = Chem.GetSymmSSSR(targetmol)
#     count_N = 0
#     count_O = 0
#     count_C = 0
#     count_x = 0
#     count_r = 0
#     count_nei=0
#     for item in atoms:
#         a = item.GetAtomicNum()
#         x = item.GetDegree()
#         r = item.IsInRing()
#         nei = item.GetNeighbors()
#         if a == 7:
#             count_N += 1
#         if a == 6:
#             count_C += 1
#         if a == 8:
#             count_O += 1
#         if x >= 3:
#             count_x += 1
#         if r :
#             count_r +=1
#         if len(nei)>=3:
#             count_nei +=1
#     if count_N >1 or count_O > 1:
#         return(-2)
#     if len(ssr) >= 1:
#         return(-2)
#     elif count_C == 5 :
#         return (1)
#     elif count_C == 6 :
#         return (1.5)
#     elif count_C == 7:
#         return (3)
#     elif count_C == 8 :
#         return (4)
#     elif count_C == 9 :
#         return (5)
#     elif count_C == 10:
#         return (2)
#     elif count_C == 11:
#         return (1)
#     # elif count_C <= 11:
#     #     return (count_C/2)
#     # elif count_C > 9:
#     #     return (0.1)
#     # elif count_C < 7:
#     #     return (0.1)
#     if count_C > 11:
#         return (-2)
#     # if count_nei >= 2:
#     # # if count_r >= 5:
#     #     print(count_nei)
#     #     return (2+(count_nei)*0.5)
#     else:
#         return(0)

def atom_num_calculate(input):
    targetmol = Chem.MolFromSmiles(input)
    atoms = targetmol.GetAtoms()
    ssr = Chem.GetSymmSSSR(targetmol)
    charge_num =Chem.rdmolops.GetFormalCharge(targetmol)
    count_N = 0
    count_O = 0
    count_C = 0
    count_F = 0
    count_cl = 0
    count_S = 0
    count_Br = 0
    count_x = 0
    count_r = 0
    count_nei=0
    for item in atoms:
        a = item.GetAtomicNum()
        x = item.GetDegree()
        r = item.IsInRing()
        nei = item.GetNeighbors()
        if a == 7:
            count_N += 1
        if a == 6:
            count_C += 1
        if a == 8:
            count_O += 1
        if a == 9:
            count_F += 1
        if a == 13:
            count_cl += 1
        if a == 15:
            count_S += 1
        if a == 35:
            count_Br += 1
        if x >= 3:
            count_x += 1
        if r :
            count_r +=1
        if len(nei)>=3:
            count_nei +=1
    if count_F >0 or count_cl >0 or count_S >0 or count_Br>0  :
        return(-5)
    if count_N >1 or count_O > 1:
        return(-3)
    if len(ssr) > 1:
        return(-3)
    # elif count_C == 5 :
    #     return (1)
    # elif count_C == 6 :
    #     return (1.5)
    # elif count_C == 7:
    #     return (3)
    # elif count_C == 8 :
    #     return (4)
    # elif count_C == 9 :
    #     return (5)
    # elif count_C == 10:
    #     return (2)
    # elif count_C == 11:
    #     return (1)
    # if count_C <= 11:
    #     return (count_C/2)
    # elif count_C > 9:
    #     return (0.1)
    # elif count_C < 7:
    #     return (0.1)
    # if charge_num != 1:
    #     return(-3)
    if count_C > 11:
        return (-3)
    # if len(ssr) == 1:
    #     return(3)
    # if count_nei >= 2:
    # # if count_r >= 5:
    #     print(count_nei)
    #     return (2+(count_nei)*0.5)
    else:
        return(0)

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)
