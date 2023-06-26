import subprocess
import platform
import os
import re
import argparse
import torch
import torch.nn as nn
from torch.autograd import Function
from collections import namedtuple
import torch.nn.functional as F
from argparse import Namespace
from typing import List, Union
import numpy as np
from torch.autograd import Variable
from chemprop.features import BatchMolGraph, mol2graph
from chemprop.features import BatchSmilesSquence,smile2smile
from chemprop.features import construct_seq_index,get_smiles_feature

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        torch.nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pad_2d_unsqueeze(x, padlen):
    #x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)



class SEQencoder(torch.nn.Module):

    def __init__(self, args):
        super(SEQencoder, self).__init__()

        self.args = args
        self.input_dim = args.seq_input_dim
        self.hidden_size = args.seq_hidden_size
        self.latent_size = args.seq_latent_size
        self.dropout = args.seq_dropout
        self.layer = args.seq_layer
        self.seq_index= construct_seq_index()
        self.encoder = torch.nn.GRU(self.input_dim, int(self.hidden_size / 2), self.layer, batch_first=True, bidirectional=True, dropout=self.dropout)

        self.AtomEmbedding = torch.nn.Embedding(len(self.seq_index),
                                                 self.hidden_size)

        self.AtomEmbedding.weight.requires_grad = True


        self.apply(weights_init)

    def forward(self, smile_list: BatchSmilesSquence, features_batch=None) -> torch.FloatTensor:
        smile_batch = smile_list.get_components()
        smile_feature,smile_sequence = get_smiles_feature(smile_batch,self.seq_index)


        # print(last_hidden.size())


        seq_vecs = []

        for sequence in smile_sequence:

            smile_emb = self.AtomEmbedding(sequence)
            smile_emb = smile_emb.reshape(1, -1, self.input_dim)
            smile_embbeding, last_hidden = self.encoder(smile_emb)

            smile_embbeding = smile_embbeding.squeeze(0)
            seq_vecs.append(smile_embbeding.mean(0))

        seq_vecs = torch.stack(seq_vecs, dim=0)


        return seq_vecs,0,0,0

    def _initialize_hidden_state(self, batch_size):
        if torch.cuda.is_available():
            return torch.zeros(self.layer * 2, 1, int(self.hidden_size/2)).cuda()
        else:
            return torch.zeros(self.layer * 2, 1, int(self.hidden_size/2))


class Seq_enconder(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(Seq_enconder, self).__init__()
        self.args = args

        #self.encoder = encoder()
        # self.atom_fdim = atom_fdim or get_atom_fdim(args)
        # self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
        #                     (not args.atom_messages) * self.atom_fdim # * 2
        self.graph_input = graph_input
        self.encoder = SEQencoder(self.args)
        self.max_seq_count = 200

    def forward(self, batch: Union[List[str], BatchSmilesSquence],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = smile2smile(batch, self.args)
        output,mask_embbeding,x,max_seq_count = self.encoder.forward(batch, features_batch)

        return output,mask_embbeding,x,max_seq_count
