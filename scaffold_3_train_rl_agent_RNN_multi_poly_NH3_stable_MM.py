#!/usr/bin/env python
import argparse
from chemprop.parsing import parse_train_args, modify_train_args, parse_predict_args, modify_predict_args
from chemprop.train.make_predictions import make_predictions_scaffold

import warnings

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from rdkit import Chem
from torch.utils.data import DataLoader
from models.model_rnn_scaffold import RNN
import matplotlib.pyplot as plt
# from utils.data_structs import Vocabulary, Experience
# from utils.data_structs_distill_scaffold import Vocabulary, Experience
from utils.data_structs_scaffold import MolData_sample_molecule, Vocabulary, Experience_scaffold
from utils.properties import get_scoring_function, qed_func_scaffold, sa_func, multi_scoring_functions_one_hot,sim_calculate,ring_calculate,atom_num_calculate
from utils.utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, fraction_valid_smiles_scaffold
from models.model_generator import transformer_RL
import utils.scaffold_utils.chem as uc
# import utils.scaffold_utils.scaffold_ring as usc
import utils.scaffold_utils.scaffold as usc

from utils.utils import decrease_learning_rate
from feature_calculate import feature_calculation, feature_same_calculation
from feature_calculate_multi import feature_calculation_HC10, feature_calculation_Ecoli, feature_calculation_Saureus
from score_transform import ComponentSpecificParametersEnum, TransformationTypeEnum, TransformationFactory, render_curve
import seaborn as sns
import os
import math

warnings.filterwarnings("ignore")
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
tt_enum = TransformationTypeEnum()
csp_enum = ComponentSpecificParametersEnum()
factory = TransformationFactory()


def score_tranform_HC10(value):
    values_list = np.arange(-1, 12, 0.25).tolist()

    specific_parameters = {csp_enum.TRANSFORMATION: True,
                           csp_enum.LOW: 0,
                           csp_enum.HIGH: 7.5,
                           csp_enum.K: 0.3,
                           csp_enum.TRANSFORMATION_TYPE: tt_enum.SIGMOID}
    transform_function = factory.get_transformation_function(specific_parameters)
    transformed_scores = transform_function(predictions=value,
                                            parameters=specific_parameters)
    return transformed_scores


def poly_unit_end(sca_1):
    ens_sca_1 = list(sca_1)
    ens_sca_1.insert(1, '1')
    ens_sca_1.insert(-4, '1')
    x = "".join(ens_sca_1)
    return x


def polymer_to_bigsmiles(polymer_1, polymer_2):
    bigsmiles_list = []
    start = 'c1cc(C(C)(C)C)ccc1C(=O)'
    if polymer_1 == None or polymer_2 == None:
        bigsmiles_list.append(None)
    elif polymer_1 == polymer_2 and polymer_1 == polymer_1:
        bigsmiles = '{' + '[>]' + polymer_1 + '.[+rn=5],' + polymer_1 + '[<].[+rn=5]}'
        bigsmiles_list.append(bigsmiles)
    else:
        for i in range(1, 10):
            bigsmiles = '{' + '[>]' + polymer_1 + '.[+rn=' + str(10*i) + '],' + polymer_2 + '[<]' + '.[+rn=' + str(
                10*(10 - i)) + ']}'
            bigsmiles_list.append(bigsmiles)
    # print(bigsmiles_list,6667777)
    return bigsmiles_list


def generate_randomized_not_repeated(
        smi, num_rand,
        max_rand):
    mol = uc.to_mol(smi)
    randomized_scaffolds = set()
    for _ in range(max_rand):
        randomized_scaffolds.add(usc.to_smiles(mol, variant="random"))
        # randomized_scaffolds.add(usc.to_smiles(mol))

        if len(randomized_scaffolds) == num_rand:
            break
    return list(randomized_scaffolds)


def prepare(input_scaffold_path):
    # input_scaffold_path = './data/target_scaffold_4.csv'
    input_scaffolds = list(uc.read_smi_file(input_scaffold_path))
    list_origin_scaffold = []
    list_scaffold_with_num = []
    list_attach_point = []
    list_randomlized_smiels = []
    list_uncanonicalized_smiels = []
    for smi in input_scaffolds:
        uncanonicalized_scaffold = usc.add_attachment_point_numbers_uncan(smi)
        # uncanonicalized_scaffold = smi
        randomized_scaffold_udf = generate_randomized_not_repeated(uncanonicalized_scaffold, num_randomized_smiles,
                                                                   max_randomized_smiles_sample)  # 生成randomlized 分子

        for smil in randomized_scaffold_udf:
            list_origin_scaffold.append(smi)
            list_uncanonicalized_smiels.append(uncanonicalized_scaffold)
            list_scaffold_with_num.append(smil)

            attachment_points = usc.get_attachment_points(smil)
            list_attach_point.append(attachment_points)

            smiles = usc.remove_attachment_point_numbers(smil)
            list_randomlized_smiels.append(smiles)

    return list_uncanonicalized_smiels, list_scaffold_with_num, list_randomlized_smiels, list_attach_point


def train_agent_func_test_args(arg_dict, input_scaffold_path):
    restore_prior_from = arg_dict.restore_prior_from
    restore_agent_from = arg_dict.restore_agent_from
    agent_save = arg_dict.agent_save
    n_steps = arg_dict.n_steps
    batch_size = arg_dict.batch_size_rl
    sigma = arg_dict.sigma
    experience_replay = 0
    save_dir = './MCMG_results/'
    voc = Vocabulary(init_from_file="data/Voc_RE1")

    start_time = time.time()

    Prior = RNN(num_layers, d_model, voc, trans_dropout)
    Agent = RNN(num_layers, d_model, voc, trans_dropout)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.decorator.load_state_dict(torch.load(restore_prior_from, map_location={'cuda:1': 'cuda:1'}))
        Agent.decorator.load_state_dict(torch.load(restore_agent_from, map_location={'cuda:1': 'cuda:1'}))
    else:
        Prior.decorator.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.decorator.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.decorator.parameters():
        param.requires_grad = False
    # for param in Agent.decorator._encoder.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.decorator.parameters(), lr=0.0001)  #bat 30 0.001

    experience = Experience_scaffold(voc)

    print("Model initialized, starting training...")

    # Scoring_function
    scoring_function1 = get_scoring_function('atom_num')
    # scoring_function1 = get_scoring_function('logP')######
    # scoring_function2 = get_scoring_function('syntax')
    expericence_step_index = []
    smiles_save = []
    score_save = []
    scaffold_smiles_save = []
    ori_scaffold_save = []
    attach_point_save = []
    final_save = []
    final_activity = []
    plot_reward = []
    plot_HC10 = []
    plot_atom = []

    for step in range(n_steps):

        list_uncanonicalized_smiels, list_scaffold_with_num, list_randomlized_smiels, list_attach_point = prepare(
            input_scaffold_path)
        dataset = MolData_sample_molecule(list_uncanonicalized_smiels, list_scaffold_with_num, list_randomlized_smiels,
                                          list_attach_point, voc)
        generate_data = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=MolData_sample_molecule.collate_fn, drop_last=True)
        # Sample from Agent
        for step_, batch in tqdm(enumerate(generate_data), total=len(generate_data)):
            scaffold_seqs = batch[0].long()
            ori_scaffold_seqs = batch[1]
            attach_points = batch[3]
            seqs, agent_likelihood, entropy = Agent.sample_rl(batch_size, scaffold_seqs, max_length=140)
            # scaffold_list = seq_to_smiles(scaffold_seqs, voc)

            # scaffold = seq_to_smiles(scaffold_seqs, voc)
            # smiles = seq_to_smiles(seqs, voc)
            # generated_smiles = []
            # for sca, dec in zip(scaffold , smiles):
            #     mol = usc.join_joined_attachments(sca, dec)
            #     if mol =
            #     generated_smiles.append(Chem.MolToSmiles(mol))
            # print(generated_smiles)
            # print('*********************')
            # unique_idxs = unique(generated_smiles)

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]
            scaffold_list = scaffold_seqs[unique_idxs]
            ori_scaffold_list = [ori_scaffold_seqs[np.array(unique_idxs)[i]] for i in range(len(unique_idxs))]
            attach_points_list = [attach_points[np.array(unique_idxs)[i]] for i in range(len(unique_idxs))]

            # Get prior likelihood and score
            prior_likelihood = Prior.likelihood_rl_reward(scaffold_list, seqs)  # 可以改？ 用Trans算？

            scaffold = seq_to_smiles(scaffold_list, voc)
            smiles = seq_to_smiles(seqs, voc)
            # print(ori_scaffold_list,111)
            # print(scaffold,222)
            # print(smiles)
            ##################################################################
            polymer_1_list = []
            polymer_2_list = []
            count_polymer_1 = []
            count_polymer_2 = []
            polymer_1_unit = []
            polymer_2_unit = []
            end_scaffold_1 = []
            end_scaffold_2 = []
            for sca in ori_scaffold_list:
                regex = '(\[[^\[\]]{1,6}\])'
                x = sca.rsplit('N')
                # print(x)
                tokenized_1 = []
                tokenized_2 = []
                if len(x) == 2:
                    polymer_1 = 'N[C@@H](C)[C@@H](C[NH3+])C(=O)'
                    polymer_2 = 'N' + x[-1]
                    polymer_1_list.append(polymer_1)
                    polymer_2_list.append(polymer_2)
                    end_scaffold_1.append(poly_unit_end(polymer_1))
                    end_scaffold_2.append(poly_unit_end(polymer_2))
                    char_list_1 = re.split(regex, polymer_1)
                    char_list_2 = re.split(regex, polymer_2)
                    for char in char_list_1:
                        if char.startswith('[*:'):
                            tokenized_1.append(char.split('[*:')[1].split(']')[0])
                    for char in char_list_2:
                        if char.startswith('[*:'):
                            tokenized_2.append(char.split('[*:')[1].split(']')[0])
                elif len(x) == 1:
                    polymer_1 = 'N[C@@H](C)[C@@H](C[NH3+])C(=O)'
                    polymer_2 = 'N' + x[-1]
                    polymer_1_list.append(polymer_1)
                    polymer_2_list.append(polymer_2)
                    end_scaffold_1.append(poly_unit_end(polymer_1))
                    end_scaffold_2.append(poly_unit_end(polymer_2))
                    char_list_1 = re.split(regex, polymer_1)
                    char_list_2 = re.split(regex, polymer_2)
                    for char in char_list_1:
                        if char.startswith('[*:'):
                            tokenized_1.append(char.split('[*:')[1].split(']')[0])
                    for char in char_list_2:
                        if char.startswith('[*:'):
                            tokenized_2.append(char.split('[*:')[1].split(']')[0])
                elif len(x) == 3:
                    polymer_1 = 'N' + x[1]
                    polymer_2 = 'N' + x[-1]
                    polymer_1_list.append(polymer_1)
                    polymer_2_list.append(polymer_2)
                    end_scaffold_1.append(poly_unit_end(polymer_1))
                    end_scaffold_2.append(poly_unit_end(polymer_2))
                    char_list_1 = re.split(regex, polymer_1)
                    char_list_2 = re.split(regex, polymer_2)
                    for char in char_list_1:
                        if char.startswith('[*:'):
                            tokenized_1.append(char.split('[*:')[1].split(']')[0])
                    for char in char_list_2:
                        if char.startswith('[*:'):
                            tokenized_2.append(char.split('[*:')[1].split(']')[0])
                elif len(x) == 4:
                    polymer_1 = 'N' + x[1] + 'N' + x[2]
                    polymer_2 = 'N' + x[-1]
                    polymer_1_list.append(polymer_1)
                    polymer_2_list.append(polymer_2)
                    end_scaffold_1.append(poly_unit_end(polymer_1))
                    end_scaffold_2.append(poly_unit_end(polymer_2))
                    char_list_1 = re.split(regex, polymer_1)
                    char_list_2 = re.split(regex, polymer_2)
                    for char in char_list_1:
                        if char.startswith('[*:'):
                            tokenized_1.append(char.split('[*:')[1].split(']')[0])
                    for char in char_list_2:
                        if char.startswith('[*:'):
                            tokenized_2.append(char.split('[*:')[1].split(']')[0])
                count_polymer_1.append(tokenized_1)
                count_polymer_2.append(tokenized_2)

            # print(polymer_1_list,111)
            # print(polymer_2_list,222)
            # print(end_scaffold_1,111)
            # print(end_scaffold_2,222)
            # print(count_polymer_1,111)
            # print(count_polymer_2,222)
            decorators = []
            for dec in smiles:
                x = dec.rsplit('|')
                decorators.append(x)
            # print(decorators, 111)
            ##################################################################
            for scaffold_1, scaffold_2, count_1, count_2, decorator, attach_points, end_scaffold_1, end_scaffold_2 in zip(
                    polymer_1_list, polymer_2_list, count_polymer_1, count_polymer_2, decorators, attach_points_list,
                    end_scaffold_1, end_scaffold_2):
                sca_1 = usc.remove_attachment_point_numbers(scaffold_1)
                sca_2 = usc.remove_attachment_point_numbers(scaffold_2)
                # print(len(attach_points))
                if len(decorator) == len(attach_points):
                    for num in count_1:
                        dec_for_sca = decorator[attach_points.index(int(num))]
                        polymer_1 = usc.join_first_attachment(sca_1,dec_for_sca)
                        # polymer_1_end = usc.join_first_attachment(end_1, dec_for_sca)
                        if polymer_1 == None:
                            polymer_1_unit.append(None)
                            # end_unit_1.append(None)
                            break
                        sca_1 = Chem.MolToSmiles(polymer_1)
                        # end_1 = Chem.MolToSmiles(polymer_1_end)
                    if polymer_1 != None:
                        polymer_1_unit.append((Chem.MolToSmiles(polymer_1)))
                    else:  ##############
                        polymer_1_unit.append(None)##############
                    # if polymer_1_end != None:
                        # end_unit_1.append((Chem.MolToSmiles(polymer_1_end)))

                    # polymer_1_unit.append(scaffold_1)
                    # polymer_1_unit.append('N[C@@H](C)[C@@H](C[NH3+])C(=O)')
                    # end_unit_1.append(None)

                    for num in count_2:
                        dec_for_sca = decorator[attach_points.index(int(num))]
                        # if usc.to_smiles(uc.to_mol(dec_for_sca))!= None:
                        #     dec_for_sca = usc.to_smiles(uc.to_mol(dec_for_sca))
                        polymer_2 = usc.join_first_attachment(sca_2, dec_for_sca)
                        # polymer_2_end = usc.join_first_attachment(end_2, dec_for_sca)
                        if polymer_2 == None:
                            polymer_2_unit.append(None)
                            # end_unit_2.append(None)
                            break
                        sca_2 = Chem.MolToSmiles(polymer_2)
                        # end_2 = Chem.MolToSmiles(polymer_2_end)
                    if polymer_2 != None:
                        polymer_2_unit.append((Chem.MolToSmiles(polymer_2)))
                    # else:##############
                    #     polymer_2_unit.append(None)##############
                    # if polymer_2_end != None:
                    #     end_unit_2.append((Chem.MolToSmiles(polymer_2_end)))
                else:
                    polymer_1_unit.append(None)
                    polymer_2_unit.append(None)
                    # end_unit_1.append(None)
                    # end_unit_2.append(None)

            ##################################################################
            # print(len(polymer_1_unit),polymer_1_unit)
            # print(len(polymer_2_unit),polymer_2_unit)
            # print(len(end_unit_1),end_unit_1)
            # print(len(end_unit_2),end_unit_2)
            ##################################################################

            bigsmiles_list = []
            feature_list = []
            array_featue_list = []
            sim_value_list = []
            ring_value_list = []
            atom_value_list = []

            for poly_1, poly_2 in zip(polymer_1_unit, polymer_2_unit):
                # print(poly_1,poly_2)
                bigsmiles = polymer_to_bigsmiles(poly_1, poly_2)
                # print(bigsmiles)
                bigsmiles_list.append(bigsmiles)

            for poly_1, poly_2 in zip(polymer_1_unit, polymer_2_unit):
                # print(poly_1,poly_2)
                if poly_1 == None or poly_2 == None or (poly_1 == poly_2):
                    feature_list.append(None)
                    array_featue_list.append(None)
                    atom_value_list.append(None)
                    sim_value_list.append(None)
                    ring_value_list.append(None)
                elif Chem.MolFromSmiles(poly_2) == None or Chem.MolFromSmiles(poly_1) == None:
                    feature_list.append(None)
                    array_featue_list.append(None)
                    atom_value_list.append(None)
                    sim_value_list.append(None)
                    ring_value_list.append(None)
                elif poly_1 == poly_2:
                    polymer_feature, array_list = feature_same_calculation(Chem.MolFromSmiles(poly_1),
                                                                           Chem.MolFromSmiles(poly_2))
                    sim_value = sim_calculate(poly_2)
                    ring_value= ring_calculate(poly_2)
                    atom_value = atom_num_calculate(poly_2)
                    atom_value_list.append(atom_value)
                    feature_list.append(polymer_feature)
                    array_featue_list.append(array_list)
                    sim_value_list.append(sim_value)
                    ring_value_list.append(ring_value)


                else:
                    # print(poly_1,poly_2,111)
                    # print(Chem.MolFromSmiles(poly_1), Chem.MolFromSmiles(poly_2), 111)
                    # polymer_feature,array_list = feature_calculation(Chem.MolFromSmiles(poly_1), Chem.MolFromSmiles(poly_2))
                    polymer_feature, array_list = feature_calculation_HC10(Chem.MolFromSmiles(poly_1),
                                                                           Chem.MolFromSmiles(poly_2))
                    sim_value = sim_calculate(poly_2)
                    ring_value= ring_calculate(poly_2)
                    atom_value = atom_num_calculate(poly_2)

                    feature_list.append(polymer_feature)
                    array_featue_list.append(array_list)
                    sim_value_list.append(sim_value)
                    ring_value_list.append(ring_value)
                    atom_value_list.append(atom_value)

            print(len(bigsmiles_list), len(feature_list))
            print(atom_value_list)
            ##################################################################
            scores = []
            scores_HC10=[]
            scores_atom = []
            activity = []
            count = 0
            # print(len(bigsmiles_list),len(feature_list),len(array_featue_list))
            for bigsmile, feature, array,sim_val,ring_val,atom_val in zip(bigsmiles_list, feature_list, array_featue_list,sim_value_list,ring_value_list,atom_value_list):
                # print(bigsmile[0])
                if bigsmile[0] == None or feature == None :
                    # print(bigsmile[0], count)
                    scores.append(-5)
                    scores_HC10.append(-1)
                    scores_atom.append(-4)
                    activity.append(-100)
                    count += 1
                else:
                    # print(bigsmile[0], count)
                    # print(bigsmile, 444)
                    pred, smiles_xx, feature = make_predictions_scaffold(args=arg_dict, smiles=bigsmile,
                                                                         features=feature, array=array)
                    # print(pred, 333)
                    # value = pred.sum()/len(pred)
                    value = pred.max()
                    # print(value,count)

                    # if value >15:
                    #     scores.append(-0.1)
                    #     activity.append(1.573 * np.exp(0.6923 * value/2.3)/6)
                    #     count += 1
                    # elif value < -1:
                    #     scores.append(-0.1)
                    #     activity.append(1.573 * np.exp(0.6923 * value/2.3)/6)
                    #     count += 1
                    # else :
                    #     # scores.append(10-1.305 *np.exp(0.7301*value/2.3))
                    #     activity.append(1.305 * np.exp(0.7301 * value))
                    #     scores.append(1.573 * np.exp(0.6923 * abs(value)/2.3)/6)
                    #     count += 1

                    if value < -1:
                        score_HC10 = [-0.1]
                        HC10 = 1.573 * np.exp(0.6923 * value)
                    elif value > 12:
                        score_HC10 = [0.1]
                        HC10 = 1.573*np.exp(0.6923*value)
                    else:
                        score_HC10 = score_tranform_HC10([abs(value)])
                        HC10 = 1.573 * np.exp(0.6923 * value)
                    scores.append(5 * score_HC10[0] + atom_val)
                    scores_HC10.append(score_HC10[0])
                    # scores.append(2*score_HC10[0]#+atom_val)
                    # scores.append(atom_val)
                    # scores.append(5 * score_HC10[0])
                    activity.append(HC10)
                    scores_atom.append(atom_val)
                    count += 1

            # print(len(scores),99999)
            ##################################################################

            # score1 = scoring_function1(scaffold,smiles)
            # score2 = scoring_function2(scaffold,smiles)
            # qed = qed_func_scaffold()(scaffold,smiles)
            # sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
            #               dtype=np.float32)  # to keep all reward components between [0,1]
            # score = scores# + score2
            score = np.float32(scores)
            sum_score = score.sum() / len(score)

            score_HC10 = np.float32(scores_HC10)
            sum_score_HC10 = score_HC10.sum() / len(scores_HC10)

            # scores_Saureus = np.float32(scores_Saureus)
            # sum_score_Saureus = scores_Saureus.sum() / len(scores_Saureus)
            #
            # scores_Ecoli = np.float32(scores_Ecoli)
            # sum_score_Ecoli = scores_Ecoli.sum() / len(scores_Ecoli)

            scores_atom = np.float32(scores_atom)
            sum_score_atom = scores_atom.sum() / len(scores_atom)

            # 判断是否为success分子，并储存
            # success_score = multi_scoring_functions_one_hot(smiles, ['drd2', 'logP'])
            # itemindex = list(np.where(success_score == 3))
            itemindex = list(np.where(score >= 0))
            # itemindex = list(np.where(score_HC10 > 0.1))
            success_smiles = list(np.array(smiles)[itemindex])
            success_smiles_scaffold = list(np.array(scaffold)[itemindex])
            success_smiles_ori_scaffold = list(np.array(ori_scaffold_list)[itemindex])
            success_smiles_attach_points = list(np.array(attach_points_list)[itemindex])
            success_smiles_score = list(np.array(score)[itemindex])
            success_prior_likehood = prior_likelihood[itemindex]
            success_bigsmiles_list = list(np.array(bigsmiles_list)[itemindex])
            success_activity =list(np.array(activity)[itemindex])

            smiles_save.extend(success_smiles)
            scaffold_smiles_save.extend(success_smiles_scaffold)
            ori_scaffold_save.extend(success_smiles_ori_scaffold)
            attach_point_save.extend(success_smiles_attach_points)
            score_save.extend(success_smiles_score)
            final_save.extend(success_bigsmiles_list)
            final_activity.extend(success_activity)
            expericence_step_index = expericence_step_index + len(success_smiles) * [step]

            if step % 10 == 0 and step != 0:
                # if step >= n_steps - 1:
                print('num: ', len(set(smiles_save)))
                print(len(smiles_save),len(scaffold_smiles_save),len(ori_scaffold_save),len(final_save),len(score_save),len(final_activity),len(attach_point_save))
                save_smiles_df = pd.concat(
                    [pd.DataFrame(smiles_save), pd.DataFrame(scaffold_smiles_save), pd.DataFrame(ori_scaffold_save),
                     pd.DataFrame(final_save), pd.DataFrame(score_save), pd.DataFrame(final_activity),
                     pd.DataFrame(attach_point_save)], axis=1)
                # save_smiles_df = pd.DataFrame(smiles_save,scaffold_smiles_save)
                save_smiles_df.to_csv(save_dir + str(step) + '_MCMG_drd.csv', index=False, header=False)
                break
            if step % 20 == 0 and step != 0:
                torch.save(Agent.decorator.state_dict(), agent_save)

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood + sigma * Variable(score)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if experience_replay and len(experience) > 4:
                exp_seqs, exp_score, exp_prior_likelihood, exp_scaffold = experience.sample(4)
                exp_agent_likelihood = Agent.likelihood_rl_reward(exp_scaffold, exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

            # Then add new experience
            # if step >50:
            #     success_prior_likehood = success_prior_likehood.data.cpu().numpy()
            #     new_experience = zip(success_smiles, success_smiles_score, success_prior_likehood,success_smiles_scaffold)
            # if step <=50:
            #     prior_likelihood = prior_likelihood.data.cpu().numpy()
            #     new_experience = zip(smiles, score, prior_likelihood,scaffold)

            # success_prior_likehood = success_prior_likehood.data.cpu().numpy()
            # new_experience = zip(success_smiles, success_smiles_score, success_prior_likehood, success_smiles_scaffold)

            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles, score, prior_likelihood,scaffold)
            experience.add_experience(new_experience)
            print(str(len(experience)) + '**********************')

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if step % 10 == 0 and step != 0:
            # decrease_learning_rate(optimizer, decrease_by=0.03)

            # Convert to numpy arrays so that we can print them
            augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            agent_likelihood = agent_likelihood.data.cpu().numpy()

            # Print some information for this step
            time_elapsed = (time.time() - start_time) / 3600
            time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
            print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
                step, fraction_valid_smiles_scaffold(scaffold, smiles) * 100, time_elapsed, time_left))
            print("  Agent    Prior   Target   Score             SMILES")
            for i in range(5):
                print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                           prior_likelihood[i],
                                                                           augmented_likelihood[i],
                                                                           score[i],
                                                                           smiles[i]))
            print("total score {}".format(sum_score))
            plot_reward.append(sum_score)
            plot_HC10.append(sum_score_HC10)
            # plot_Saureus.append(sum_score_Saureus)
            # plot_Ecoli.append(sum_score_Ecoli)
            plot_atom.append(sum_score_atom)
            # if step % 10 == 0 and step != 0:
            name_1 = ['reward']
            if step % 15 == 0:

                plt.plot(plot_reward,label='Reward')
                # plt.plot(plot_HC10,label='HC10')
                # plt.plot(plot_Saureus,label='Saureus')
                # plt.plot(plot_Ecoli,label='Ecoli')
                plt.plot(plot_atom,label='atom')
                plt.legend()
                plt.xlabel('Training iteration')
                plt.ylabel('Average reward')
                plt.savefig('./MCMG_results/' + str(step) + '_reward')
                plt.draw()
                plt.pause(1)  # 间隔的秒数： 1s
                plt.close()


                reward_list = pd.DataFrame(plot_reward,columns=name_1)
                reward_list.to_csv('./MCMG_results/reward_list.csv', header=True, index=True)


if __name__ == "__main__":
    d_model = 512  # 128
    num_layers = 3
    trans_dropout = 0
    max_seq_length = 140

    num_epochs = 50  # 600
    n_steps = 5  # 5000

    num_randomized_smiles =3   # 14
    num_decorations_per_scaffold = 1
    max_randomized_smiles_sample = 10000  # 10000

    input_scaffold_path = './data/target_scaffold_binary.csv'

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=1010)
    parser.add_argument('--batch-size', action='store', dest='batch_size_rl', type=int,
                        default=25)  #30 good
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=60)

    parser.add_argument('--middle', action='store', dest='restore_prior_from',
                        # default='./ckpt/scaffold_middle_13w__target_dim512_lr1e-9.ckpt',
                        default='./ckpt/scaffold_middle_13w_target_dim512_lr1e-9_third.ckpt',
                        # default='./ckpt/CNO_scaffold_middle_lr1e-9_NH3_7.19_best.ckpt', # important
                        # default='./ckpt/CNO_scaffold_middle_lr1e-9_NH3+_7.21_best.ckpt',
                        # default='./ckpt/scaffold_middle_13w_target_dim512_lr1e-9_NH3_6.14.ckpt',
                        help='Path to an RNN checkpoint file to use as a Prior')

    parser.add_argument('--agent', action='store', dest='restore_agent_from',
                        # default='./ckpt/scaffold_middle_13w__target_dim512_lr1e-9.ckpt',
                        default='./ckpt/scaffold_middle_13w_target_dim512_lr1e-9_third.ckpt',
                        # default='./ckpt/CNO_scaffold_middle_lr1e-9_NH3_7.19_best.ckpt', # important
                        # default='./ckpt/CNO_scaffold_middle_lr1e-9_NH3+_7.21_best.ckpt',
                        # default='./ckpt/scaffold_middle_13w_target_dim512_lr1e-9_NH3_6.14.ckpt',
                        help='Path to an RNN checkpoint file to use as a Agent.')

    parser.add_argument('--save-file-path', action='store', dest='agent_save', default='./data/RL_agent.ckpt',
                        help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    # arg_dict = vars(parser.parse_args())
    # print(arg_dict)

    ##############################
    ######predict parameters######

    # parser.checkpoint_dir = './ckpt/gra+seq no feature 150 lr 1e-4 multi 1 8 1 1'
    parser = parse_train_args(parser)

    arg_dict = parser.parse_args()

    # arg_dict.checkpoint_dir = './ckpt/5.31 gaijin trans/choose'
    # arg_dict.checkpoint_dir = './ckpt/Feature+LOGP/15'
    # arg_dict.checkpoint_dir = './ckpt/HC10_feature_only_HC10split/10'
    arg_dict.checkpoint_dir = './ckpt/HC10_all_feature_only/choose_2'
    # arg_dict.checkpoint_dir = './ckpt/10'

    modify_train_args(arg_dict)
    print(arg_dict)

    # prepare()

    # train_agent_func_test(**arg_dict)

    train_agent_func_test_args(arg_dict, input_scaffold_path)
