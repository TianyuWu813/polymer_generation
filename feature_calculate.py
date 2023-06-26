
name = ['nBase', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS6s', 'AATS6d', 'AATS6s', 'AATS6Z', 'AATS0m', 'AATS1m',
        'AATS2m', 'AATS3m', 'AATS4m', 'AATS6m', 'AATS0v', 'AATS3v', 'AATS6se', 'ATSC2c', 'ATSC4c', 'ATSC7c', 'ATSC7dv',
        'ATSC5d', 'ATSC4s', 'ATSC4pe', 'ATSC5p', 'AATSC4s', 'AATSC6m', 'AATSC6v', 'AATSC5i', 'AATSC6i', 'MATS3c', 'MATS4c',
        'MATS4pe', 'GATS6c', 'C3SP3', 'Xch-5d', 'Xch-6d', 'Xc-3d', 'Xc-4d', 'CIC1', 'CIC4', 'Kier3', 'PEOE_VSA9',
        'SMR_VSA4', 'SMR_VSA7', 'SlogP_VSA4', 'EState_VSA5', 'MID_h', 'MPC8', 'MPC9', 'n6Ring', 'n7Ring', 'n8Ring',
        'Diameter', 'SRW07', 'TSRW10', 'nBase.1', 'ATS5dv.1', 'ATS6dv.1', 'ATS7dv.1', 'ATS7d.1', 'ATS6s.1', 'AATS6d.1',
        'AATS6s.1', 'AATS6Z.1', 'AATS0m.1', 'AATS1m.1', 'AATS2m.1', 'AATS3m.1', 'AATS6m.1', 'AATS0v.1', 'AATS6se.1',
        'AATS6pe.1', 'ATSC2c.1', 'ATSC4c.1', 'ATSC7c.1', 'ATSC7dv.1', 'ATSC5d.1', 'ATSC4s.1', 'ATSC6v.1', 'ATSC5p.1',
        'AATSC4s.1', 'AATSC5m.1', 'AATSC6v.1', 'AATSC6i.1', 'MATS3c.1', 'MATS4c.1', 'MATS6p.1', 'GATS6c.1', 'C1SP3.1',
        'C3SP3.1', 'Xch-5d.1', 'Xch-6d.1', 'Xc-4d.1', 'Xp-5d.1', 'Xp-6d.1', 'CIC1.1', 'CIC4.1', 'Kier3.1', 'SMR_VSA4.1',
        'SMR_VSA7.1', 'SlogP_VSA4.1', 'EState_VSA5.1', 'MPC8.1', 'MPC9.1', 'n6Ring.1', 'n7Ring.1', 'n8Ring.1',
        'Diameter.1']

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


from rdkit import Chem

from mordred import Calculator, descriptors,ABCIndex
from mordred.error import Missing
import numpy as np
import  torch
from mordred.Autocorrelation import ATS
from rdkit.Chem import Descriptors

def feature_calculation(polymer_1,polymer_2):

    calc = Calculator(descriptors)
    result_polymer_1 = calc(polymer_1)
    result_polymer_2 = calc(polymer_2)
    logp_poly1 = Descriptors.MolLogP(polymer_1)
    logp_poly2 = Descriptors.MolLogP(polymer_2)

    feature_poly_1 = []
    feature_poly_2 = []

    for item in name_poly_1:
        if type(result_polymer_1[item]) == Missing:
            feature_poly_1.append(0)
        else:
            feature_poly_1.append(result_polymer_1[item])

    for item in name_poly_2:
        if type(result_polymer_2[item]) == Missing:
            feature_poly_2.append(0)
        else:
            feature_poly_2.append(result_polymer_2[item])
    tensor_list = []
    array_list = []
    for num in range(1,10):
        new_tensor_1 = []
        new_tensor_1.append(num)
        new_tensor_1.append(logp_poly1)
        new_tensor_1.extend(feature_poly_1)
        new_tensor_1.append(10-num)
        new_tensor_1.append(logp_poly2)
        new_tensor_1.extend(feature_poly_2)
        array_list.append(new_tensor_1)
        tensor_list.append(torch.tensor(new_tensor_1))
    tensor=torch.stack(tensor_list)
    return tensor,array_list

def feature_same_calculation(polymer_1,polymer_2):

    calc = Calculator(descriptors)
    result_polymer_1 = calc(polymer_1)
    result_polymer_2 = calc(polymer_2)
    logp_poly1 = Descriptors.MolLogP(polymer_1)
    logp_poly2 = Descriptors.MolLogP(polymer_2)

    feature_poly_1 = []
    feature_poly_2 = []

    for item in name_poly_1:
        if type(result_polymer_1[item]) == Missing:
            feature_poly_1.append(0)
        else:
            feature_poly_1.append(result_polymer_1[item])

    for item in name_poly_2:
        if type(result_polymer_2[item]) == Missing:
            feature_poly_2.append(0)
        else:
            feature_poly_2.append(result_polymer_2[item])
    tensor_list = []
    array_list = []
    # for num in range(1,10):
    new_tensor_1 = []
    new_tensor_1.append(5)
    new_tensor_1.append(logp_poly1)
    new_tensor_1.extend(feature_poly_1)
    new_tensor_1.append(5)
    new_tensor_1.append(logp_poly2)
    new_tensor_1.extend(feature_poly_2)
    array_list.append(new_tensor_1)
    tensor_list.append(torch.tensor(new_tensor_1))
    tensor=torch.stack(tensor_list)
    return tensor,array_list

polymer_1 = Chem.MolFromSmiles('c1ccccc1')
polymer_2 = Chem.MolFromSmiles("c1ccccc1Cl")
# polymer_feature_tendor = feature_calculate(polymer_1,polymer_2)
# polymer_feature_tendor_ex = feature_calculate(polymer_2,polymer_1)



