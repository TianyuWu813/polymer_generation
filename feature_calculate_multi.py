


name_poly_1 = ['nBase', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS6s', 'AATS6d', 'AATS6s', 'AATS6Z', 'AATS0m', 'AATS1m',
        'AATS2m', 'AATS3m', 'AATS4m', 'AATS6m', 'AATS0v', 'AATS3v', 'AATS6se', 'ATSC2c', 'ATSC4c', 'ATSC7c', 'ATSC7dv',
        'ATSC5d', 'ATSC4s', 'ATSC4pe', 'ATSC5p', 'AATSC4s', 'AATSC6m', 'AATSC6v', 'AATSC5i', 'AATSC6i', 'MATS3c','MATS4c',
        'MATS4pe', 'GATS6c', 'C3SP3', 'Xch-5d', 'Xch-6d', 'Xc-3d', 'Xc-4d', 'CIC1', 'CIC4', 'Kier3', 'PEOE_VSA9',
        'SMR_VSA4', 'SMR_VSA7', 'SlogP_VSA4', 'EState_VSA5', 'MID_h', 'MPC8', 'MPC9', 'n6Ring', 'n7Ring', 'n8Ring',
        'Diameter', 'SRW07', 'TSRW10']

name_poly_2 = ['nBase', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS7d', 'ATS6s', 'AATS6d',
        'AATS6s', 'AATS6Z', 'AATS0m', 'AATS1m', 'AATS2m', 'AATS3m', 'AATS6m', 'AATS0v', 'AATS6se',
        'AATS6pe', 'ATSC2c', 'ATSC4c', 'ATSC7c', 'ATSC7dv', 'ATSC5d', 'ATSC4s', 'ATSC6v', 'ATSC5p',
        'AATSC4s', 'AATSC5m', 'AATSC6v', 'AATSC6i', 'MATS3c', 'MATS4c', 'MATS6p', 'GATS6c', 'C1SP3',
        'C3SP3', 'Xch-5d', 'Xch-6d', 'Xc-4d', 'Xp-5d', 'Xp-6d', 'CIC1', 'CIC4', 'Kier3', 'SMR_VSA4',
        'SMR_VSA7', 'SlogP_VSA4', 'EState_VSA5', 'MPC8', 'MPC9', 'n6Ring', 'n7Ring', 'n8Ring',
        'Diameter']


name_poly_1_HC10 =['SpAbs_A','ATS4d', 'ATS0s', 'ATS1s', 'ATS4s', 'ATS5s',
                   'ATS5v', 'ATS2are', 'AATS2m', 'AATS3v', 'ATSC0d', 'ATSC6d', 'ATSC4pe',
                   'ATSC7pe', 'ATSC1i', 'ATSC6i', 'MATS6p', 'Xp-4d', 'ZMIC4', 'Kier2',]#175
name_poly_2_HC10 =[ 'ATS4d', 'ATS0s', 'ATS1s', 'ATS4s', 'ATS5s', 'ATS5v',
                   'AATS2m', 'AATS3v', 'ATSC0d', 'ATSC4pe', 'ATSC7pe',
                   'ATSC6p', 'ATSC1i', 'ATSC6i', 'MATS6p', 'Xp-4d', 'ZMIC4',
                   'Kier2', 'TSRW10']  #175

name_poly_1_Ecoli = ['ATS4d', 'ATS2are', 'AATS2m', 'AATS4m', 'AATS2v', 'ATSC6dv',
     'ATSC0d', 'ATSC5d', 'ATSC1v', 'ATSC1i', 'ATSC5i', 'AATSC1dv', 'MATS6p',
     'GATS6c', 'BCUTdv-1l', 'Xp-4d', 'ZMIC4', 'Kier2', 'MDEC-13', 'MDEC-23'] #use2

name_poly_2_Ecoli = ['ATS4d', 'AATS2m', 'AATS4m', 'AATS2v', 'AATS3v', 'ATSC1dv',
     'ATSC6dv', 'ATSC0d', 'ATSC5d', 'ATSC1v', 'ATSC1i',
     'AATSC1dv', 'AATSC6dv', 'AATSC6Z', 'MATS6p', 'GATS6c',
     'ZMIC4', 'MDEC-13', 'TSRW10'] #use2


name_poly_1_Saureus =['nAtom', 'ATS2dv', 'ATS3dv', 'ATS4d', 'ATS1s', 'ATS4s',
                   'ATS2are', 'AATS1m', 'AATS3v', 'ATSC3c', 'ATSC5d', 'ATSC0i', 'ATSC6i',
                   'AATSC6dv', 'MATS6p', 'BCUTdv-1l', 'TIC3', 'ZMIC4', 'TSRW10']

name_poly_2_Saureus= [ 'nAtom',
                   'ATS2dv', 'ATS3dv', 'ATS3d', 'ATS4d', 'ATS1s', 'AATS1m',
                   'AATS3v', 'AATS6i', 'ATSC3c', 'ATSC1dv', 'ATSC5d', 'ATSC0i',
                   'AATSC6dv', 'MATS6p', 'GATS6dv', 'TIC2', 'TIC3', 'ZMIC4',
                   'TSRW10']

from rdkit import Chem

from mordred import Calculator, descriptors,ABCIndex
from mordred.error import Missing
import numpy as np
import  torch
from mordred.Autocorrelation import ATS
from rdkit.Chem import Descriptors

def feature_calculation_HC10(polymer_1,polymer_2):

    calc = Calculator(descriptors)
    result_polymer_1 = calc(polymer_1)
    result_polymer_2 = calc(polymer_2)
    logp_poly1 = Descriptors.MolLogP(polymer_1)
    logp_poly2 = Descriptors.MolLogP(polymer_2)

    feature_poly_1 = []
    feature_poly_2 = []

    for item in name_poly_1_HC10:
        if type(result_polymer_1[item]) == Missing:
            feature_poly_1.append(0)
        else:
            feature_poly_1.append(result_polymer_1[item])

    for item in name_poly_2_HC10:
        if type(result_polymer_2[item]) == Missing:
            feature_poly_2.append(0)
        else:
            feature_poly_2.append(result_polymer_2[item])
    tensor_list = []
    array_list = []

    for num in range(1,10):
        new_tensor_1 = []
        new_tensor_1.append(int(10*num))
        new_tensor_1.append(logp_poly1)
        new_tensor_1.extend(feature_poly_1)
        new_tensor_1.append(int(10*(10-num)))
        new_tensor_1.append(logp_poly2)
        new_tensor_1.extend(feature_poly_2)
        array_list.append(new_tensor_1)
        tensor_list.append(torch.tensor(new_tensor_1))
    tensor=torch.stack(tensor_list)
    # print(tensor.size(),111) #172
    return tensor,array_list

def feature_calculation_Saureus(polymer_1,polymer_2):

    calc = Calculator(descriptors)
    result_polymer_1 = calc(polymer_1)
    result_polymer_2 = calc(polymer_2)
    logp_poly1 = Descriptors.MolLogP(polymer_1)
    logp_poly2 = Descriptors.MolLogP(polymer_2)

    feature_poly_1 = []
    feature_poly_2 = []

    for item in name_poly_1_Saureus:
        if type(result_polymer_1[item]) == Missing:
            feature_poly_1.append(0)
        else:
            feature_poly_1.append(result_polymer_1[item])

    for item in name_poly_2_Saureus:
        if type(result_polymer_2[item]) == Missing:
            feature_poly_2.append(0)
        else:
            feature_poly_2.append(result_polymer_2[item])
    tensor_list = []
    array_list = []
    for num in range(1,10):
        new_tensor_1 = []
        new_tensor_1.append(int(10*num))
        new_tensor_1.append(logp_poly1)
        new_tensor_1.extend(feature_poly_1)
        new_tensor_1.append(int(10*(10-num)))
        new_tensor_1.append(logp_poly2)
        new_tensor_1.extend(feature_poly_2)
        array_list.append(new_tensor_1)
        tensor_list.append(torch.tensor(new_tensor_1))
    tensor=torch.stack(tensor_list)
    # print(tensor.size(),222)  #176
    return tensor,array_list

def feature_calculation_Ecoli(polymer_1,polymer_2):

    calc = Calculator(descriptors)
    result_polymer_1 = calc(polymer_1)
    result_polymer_2 = calc(polymer_2)
    logp_poly1 = Descriptors.MolLogP(polymer_1)
    logp_poly2 = Descriptors.MolLogP(polymer_2)

    feature_poly_1 = []
    feature_poly_2 = []

    for item in name_poly_1_Ecoli:
        if type(result_polymer_1[item]) == Missing:
            feature_poly_1.append(0)
        else:
            feature_poly_1.append(result_polymer_1[item])

    for item in name_poly_2_Ecoli:
        if type(result_polymer_2[item]) == Missing:
            feature_poly_2.append(0)
        else:
            feature_poly_2.append(result_polymer_2[item])
    tensor_list = []
    array_list = []
    for num in range(1,10):
        new_tensor_1 = []
        new_tensor_1.append(int(10*num))
        new_tensor_1.append(logp_poly1)
        new_tensor_1.extend(feature_poly_1)
        new_tensor_1.append(int(10*(10-num)))
        new_tensor_1.append(logp_poly2)
        new_tensor_1.extend(feature_poly_2)
        array_list.append(new_tensor_1)
        tensor_list.append(torch.tensor(new_tensor_1))
    tensor=torch.stack(tensor_list)
    # print(array_list,666)
    # print(tensor.size(),333)  #158
    return tensor,array_list

polymer_1 = Chem.MolFromSmiles('c1ccccc1')
polymer_2 = Chem.MolFromSmiles("c1ccccc1Cl")




