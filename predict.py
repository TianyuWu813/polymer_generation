# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from chemprop.parsing import parse_train_args, modify_train_args,parse_predict_args,modify_predict_args
from chemprop.train import make_predictions
from chemprop.features import load_features
import torch
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
# from seaborn import sns
from rdkit.Chem import AllChem as Chem
from sklearn.preprocessing import StandardScaler

def get_fp(list_of_smi):
    """ Function to get fingerprint from a list of SMILES"""
    fingerprints = []
    mols = [Chem.MolFromSmiles(x) for x in list_of_smi]
    # if rdkit can't compute the fingerprint on a SMILES
    # we remove that SMILES
    idx_to_remove = []
    for idx, mol in enumerate(mols):
        try:
            fprint = Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
            fingerprints.append(fprint)
        except:
            idx_to_remove.append(idx)

    smi_to_keep = [smi for i, smi in enumerate(list_of_smi) if i not in idx_to_remove]
    return fingerprints, smi_to_keep


def get_embedding(data):
    """ Function to compute the UMAP embedding"""
    data_scaled = StandardScaler().fit_transform(data)

    embedding = umap.UMAP(n_neighbors=10,
                          min_dist=0.5,
                          metric='correlation',
                          random_state=16).fit_transform(data_scaled)

    return embedding


def draw_umap(embedding_hp1,pred):
    # tox_count = []
    # intox_count = []
    # for i in range(len(pred)):
    #     if pred[i][1]>0.5:
    #         tox_count.append(i)
    #     else:
    #         intox_count.append(i)
    # print(len(tox_count))
    # print(len(intox_count))

    #sns.set_theme(style="darkgrid")
    # sns.palplot(sns.color_palette("bwr", 100))
    # sns.color_palette("Reds", 10)
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.xlim([np.min(embedding_hp1[:, 0]) - 0.5, np.max(embedding_hp1[:, 0]) + 0.5])
    plt.ylim([np.min(embedding_hp1[:, 1]) - 0.5, np.max(embedding_hp1[:, 1]) + 0.5])


    colors2=[]
    for item in range(len(pred)-2):
        #print(pred[item][0])
        #x= (max-pred[item][0])/(max-min)
        p = np.array([pred[item+2][1], 1-pred[item+2][1]])
        index = np.random.choice([1,0], p=p.ravel())
        colors2.append(index)
    plt.scatter(embedding_hp1[2:, 0], embedding_hp1[2:, 1], lw=0, c=colors2,cmap='bwr', alpha=0.6, s=50, marker="o",
                 linewidth=0.5)
    plt.scatter(embedding_hp1[0, 0], embedding_hp1[0, 1], lw=0, c='Blue', alpha=1.0, s=200, marker="*",
              linewidth=0.5)
    plt.scatter(embedding_hp1[1, 0], embedding_hp1[1, 1], lw=0, c='Blue', alpha=1.0, s=200, marker="*",
                 linewidth=0.5)
    plt.setp(ax, xticks=[], yticks=[])
    fig2, ax2 = plt.subplots(figsize=(30, 30))
    #fig.colorbar(fig,cmap='RdBu')

    plt.xlim([np.min(embedding_hp1[:, 0]) - 0.5, np.max(embedding_hp1[:, 0]) + 0.5])
    plt.ylim([np.min(embedding_hp1[:, 1]) - 0.5, np.max(embedding_hp1[:, 1]) + 0.5])
    labelsize = 40
    # plt.xlabel('UMAP 1', fontsize=labelsize)
    # plt.ylabel('UMAP 2', fontsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # max = np.max(pred)
    # min = np.min(pred)
    #min = np.min(pred)
    # for item in range(len(pred)):
    #     if pred[item][0]>max:
    #        max=pred[item][0]
    #     if pred[item][0] < min:
    #         min = pred[item][0]
    # print(max,min)
    colors=[]
    for item in range(len(pred)):
        #print(pred[item][0])
        #x= (max-pred[item][0])/(max-min)
        p = np.array([pred[item][0], 1-pred[item][0]])
        index = np.random.choice([1,0], p=p.ravel())
        colors.append(index)
        #x = softmax(pred[item][1])
        # x = pred[item][1]
        # colors.append(x)
    # np.random.seed(0)
    # p = np.array([0.1, 0.0, 0.7, 0.2])
    # index = np.random.choice([0, 1, 2, 3], p=p.ravel())

    #print(colors.dtype)
    #colors = np.random.rand(642)
    #print(colors.dtype)
    print(colors)
    # for i in range(len(colors)):
    #     print(colors[i])
    #     plt.scatter(embedding_hp1[i, 0], embedding_hp1[i, 1], lw=0,c=colors[i], cmap='bwr', alpha=1.0, s=30, marker="o",
    #              linewidth=2)
    #     plt.scatter(embedding_hp1[tox_count[i], 0], embedding_hp1[tox_count[i], 1], lw=0, c=colors,cmap='Spectral', alpha=1.0, s=30, marker="o",
    #              linewidth=2)
    # for i in range(len(intox_count)):
    #     plt.scatter(embedding_hp1[intox_count[i], 0], embedding_hp1[intox_count[i], 1], lw=0, c='#00CED1', alpha=1.0, s=30, marker="o",
    #                 linewidth=2)

    plt.scatter(embedding_hp1[:, 0], embedding_hp1[:, 1], lw=0, c=colors ,cmap='bwr', alpha=0.6, s=50, marker="o",
                 linewidth=0.5)  #c=sns.color_palette("Reds",642) c='#D55E00'
    # plt.scatter(embedding_hp1[0, 0], embedding_hp1[0, 1], lw=0, c='#D55E00', alpha=1.0, s=50, marker="o",
    #              linewidth=0.5)
    # plt.scatter(embedding_hp1[1, 0], embedding_hp1[1, 1], lw=0, c='#D55E00', alpha=1.0, s=50, marker="o",
    #              linewidth=0.5)
    #leg = plt.legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
    #leg.get_frame().set_alpha(0.9)
    plt.setp(ax2, xticks=[], yticks=[])
    plt.show()



if __name__ == '__main__':

    # args = parse_predict_args()
    args = parse_train_args()
    args.checkpoint_dir = './ckpt/gra+seq no feature 150 lr 1e-4 multi 1 8 1 1'
    # modify_predict_args(args)
    modify_train_args(args)
    args.test_path= args.data_path
    # df = pd.read_csv('./data/albiciansK1_multiclass.csv')

    if args.features_path is not None:
        features_data = []
        for feat_path in args.features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None
    # pred, smiles = make_predictions(args, df.smiles.tolist())
    # print(df.smiles.tolist())
    # pred,smiles,feature = make_predictions(args, df.smiles.tolist(),features_data)
    pred,smiles,feature = make_predictions(args=args,features=features_data)
    #fp_hp1, sm_for_hp1 = get_fp(smiles_h1)
    # fp_hp1 = np.array(feature)

    # embedding_hp1 = get_embedding(feature)
    # draw_umap(embedding_hp1,pred)


    df = pd.DataFrame({'smiles':smiles})
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
    df.to_csv(f'./predict.csv', index=False)