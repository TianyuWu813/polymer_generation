# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math
import torch.nn as nn
from argparse import Namespace
import numpy as np
import torch.nn.functional as F

def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))
    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))
    def forward(self, D):
        D = D.unsqueeze(-1)
        return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

class GraphFormer(nn.Module):
    def __init__(self,n_layers,head_size,hidden_dim,dim_feedforward,attention_dropout_rate):
        super(GraphFormer, self).__init__()
        #self.save_hyperparameters()
        self.n_layers = n_layers
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.ffn_dim = dim_feedforward
        self.dropout_rate = attention_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate


        self.encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.head_size)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(self.encoders)

        self.graph_linear = nn.Linear(self.hidden_dim, self.head_size)
        self.seq_linear = nn.Linear(self.hidden_dim, self.hidden_dim // self.head_size)




    def forward(self, seq_output,graph_output):#,perturb=None):
        n_size = seq_output.size()[0]
        print(n_size)
        graph_output = self.graph_linear(graph_output)
        graph_output = graph_output.transpose(1, 2)
        # graph_output = graph_output.unsqueeze(0).repeat(n_size,1,1)
        print(graph_output.size(),12345)
        # transfomrer encoder
        # output = self.input_dropout(seq_output)
        for enc_layer in self.layers:
            output = enc_layer(seq_output, graph_output)

        return output



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.linear_atten = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.head_size, self.att_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)

        q = self.linear_q(q)#.view(batch_size, self.head_size, d_k)
        k = self.linear_k(k)#.view(batch_size, self.head_size, d_k)
        v = self.linear_v(v)#.view(batch_size, self.head_size, d_v)
        # print(attn_bias.size())


        q = q.transpose(0, 1)                  # [b, h, q_len, d_k]
        v = v.transpose(0, 1)                  # [b, h, v_len, d_v]
        k = k.transpose(0, 1).transpose(1, 2)  # [b, h, d_k, k_len]
        # print(attn_bias.size())
        # print(q.size(),1234)
        # print(k.size(), 1234)
        # print(v.size(), 1234)

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        print(x.size(),1234)
        print(attn_bias.size(), 1234)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=2)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(0, 1).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.head_size * d_v)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.head_size=head_size


        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias):
        y = self.self_attention_norm(x)
        y=y.transpose(0,1)
        y = self.self_attention(y,y,y,attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x




class Transformer(nn.Module):
    def __init__(self,n_layers,head_size,hidden_dim,dim_feedforward,attention_dropout_rate):
        super(Transformer, self).__init__()

        self.encoder = GraphFormer(n_layers,head_size,hidden_dim,dim_feedforward,attention_dropout_rate)

    def forward(self, seq_output,graph_output) -> torch.FloatTensor:
        output = self.encoder.forward(seq_output,graph_output)

        return output