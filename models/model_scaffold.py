# _*_ encoding: utf-8 _*_


import math
import numpy as np
import torch
import torch.nn as nn
# from einops import rearrange
import torch.nn.functional as F
import torch.nn.utils.rnn as tnnur
from .model_Transformer import GPTDecoderLayer, GPTDecoder
from utils.utils import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class decoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        # self.embed_src = nn.Embedding(vocab_size, d_model)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.embed_tgt_dec = nn.Embedding(vocab_size, d_model)
        self.num_layers = 2
        self.max_seq_length = max_seq_length
        self._embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Dropout(trans_dropout)
        )

        # self.pos_enc = LearnedPositionEncoding(d_model, pos_dropout, max_seq_length)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        decoder_layers = GPTDecoderLayer(d_model, nhead, dim_feedforward, trans_dropout)
        self.transformer_encoder = GPTDecoder(decoder_layers, num_decoder_layers)

        decoder_layers_dec = GPTDecoderLayer(d_model, nhead, dim_feedforward, trans_dropout)

        self.transformer_decoder = GPTDecoder(decoder_layers_dec, num_decoder_layers)

        self._attention = AttentionLayer(self.d_model)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_dec, tgt_key_padding_mask, tgt_mask, tgt_dec_key_padding_mask,tgt_dec_mask):
        # (S ,N ,EMBEDDING_DIM) 所以先要转置

        # con_token = con_token.transpose(0, 1)

        # src = src.transpose(0, 1)
        batch_size = tgt.size()[0]
        seq_length = tgt.size()[1]
        tgt_copy = tgt
        tgt_copy = tgt_copy.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_dec = tgt_dec.transpose(0,1)

        # src = rearrange(src, 'n s -> s n')
        # tgt = rearrange(tgt, 'n t -> t n')

        # src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        # add conditional token embedding

        # a = self.embed_tgt(con_token).sum(dim=0).unsqueeze(0)
        # b = self.embed_tgt(tgt)
        # c = a + b

        # tgt = self.pos_enc(
        #     (self.embed_tgt(tgt) + self.embed_tgt(tgt_dec).sum(dim=0).unsqueeze(0)) * math.sqrt(self.d_model))

        # without conditional token 可跑
        tgt = self.pos_enc(
            (self.embed_tgt(tgt)) * math.sqrt(self.d_model))

        # tgt_dec = self.pos_enc(
        #     (self.embed_tgt_dec(tgt_dec)) * math.sqrt(self.d_model))

        # tgt_dec = self.pos_enc(
        #     (self.embed_tgt_dec(tgt_dec) + self.embed_tgt(tgt_copy).sum(dim=0).unsqueeze(0)) * math.sqrt(self.d_model))

        output = self.transformer_encoder(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        tgt_dec = self.pos_enc(
            (self.embed_tgt_dec(tgt_dec) + output.sum(dim=0).unsqueeze(0)) * math.sqrt(self.d_model))

        # output = output.transpose(0, 1)
        # tgt_dec = self.embed_tgt_dec(tgt_dec)
        # seq_length = torch.tensor([tgt_dec.size()[1]]).int()
        # tgt_dec = tnnur.pack_padded_sequence(
        #     tgt_dec, seq_length.int(), batch_first=True, enforce_sorted=False)
        # print(tgt_dec.size())
        # output_dec, hidden_states = self.transformer_decoder(tgt_dec, output)
        output_dec = self.transformer_decoder(tgt_dec, tgt_mask=tgt_dec_mask, tgt_key_padding_mask=tgt_dec_key_padding_mask)

        # output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)
        output_dec =output_dec.transpose(0,1)

        mask = (output_dec[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(output_dec, output, mask)

        return self.fc(attn_padded_encoded_seqs)
        # return self.fc(output)

    def encode(self, tgt, tgt_dec, tgt_key_padding_mask, tgt_mask, tgt_dec_key_padding_mask,tgt_dec_mask):
        tgt_copy = tgt
        tgt_copy = tgt_copy.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_enc(
            (self.embed_tgt(tgt)) * math.sqrt(self.d_model))

        output = self.transformer_encoder(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        output = output.transpose(0, 1)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 可学习位置编码
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        a = weight[:x.size(0), :]
        x = x + weight[:x.size(0), :]
        return self.dropout(x)

class AttentionLayer(nn.Module):

    def __init__(self, num_dimensions):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions

        self._attention_linear = nn.Sequential(
            nn.Linear(self.num_dimensions*2, self.num_dimensions),
            nn.Tanh()
        )

    def forward(self, padded_seqs, encoder_padded_seqs, decoder_mask):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_mask: A tensor that represents the encoded input mask.
        :return : Two tensors: one with the modified logits and another with the attention weights.
        """
        # scaled dot-product
        # (batch, seq_d, 1, dim)*(batch, 1, seq_e, dim) => (batch, seq_d, seq_e*)
        attention_weights = (padded_seqs.unsqueeze(dim=2)*encoder_padded_seqs.unsqueeze(dim=1))\
            .sum(dim=3).div(math.sqrt(self.num_dimensions))\
            .softmax(dim=2)
        # (batch, seq_d, seq_e*)@(batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        attention_masked = self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2))*decoder_mask
        return (attention_masked, attention_weights)

class transformer_RL():
    def __init__(self, voc, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        self.decodertf = decoderTransformer(voc.vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
                                            max_seq_length, pos_dropout, trans_dropout)


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.decodertf.to(self.device)
        self.voc = voc
        self._nll_loss = nn.NLLLoss(ignore_index=0, reduction="none")

    def likelihood(self, target,target_decorator):
        """
               Retrieves the likelihood of a given sequence

               Args:
                   target: (batch_size * sequence_lenght) A batch of sequences

               Outputs:
                   log_probs : (batch_size) Log likelihood for each example*
                   entropy: (batch_size) The entropies for the sequences. Not
                                         currently used.
        """
        len_con_token = 0
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token_dec = Variable(torch.zeros(batch_size_dec, 1).long())

        start_token[:] = self.voc.vocab['GO']

        # split train and target and con_token
        con_token = target[:, 0:len_con_token]

        x = torch.cat((start_token, target[:, len_con_token:-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, len_con_token:-1]), 1)


        y_target = target[:, len_con_token:].contiguous().view(-1)
        y_target_dec = target_decorator[:, len_con_token:].contiguous().view(-1)



        # 去掉token长度
        seq_length = seq_length - len(con_token[0])
        seq_length_dec = seq_length_dec - len(con_token[0])

        # log_probs = Variable(torch.zeros(batch_size))
        # entropy = Variable(torch.zeros(batch_size))

        logits = Prior_train_forward(self.decodertf, x,x_dec, con_token)

        logits = F.log_softmax(logits,dim=2)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        # criterion = F.nll_loss(output.view(-1, len(vocab)),
        #                   sm.contiguous().view(-1), ignore_index=PAD)

        # log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target)
        log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target_dec)

        # loss值
        mean_log_probs = log_probs.mean()

        # print('gooo',log_probs.shape)
        log_probs = log_probs.view(-1,  seq_length_dec)

        # 求和，得到每个分子的loss值
        log_probs_each_molecule = log_probs.sum(dim=1)

        return mean_log_probs, log_probs_each_molecule

    def encode(self, target,target_decorator):
        """
               Retrieves the likelihood of a given sequence

               Args:
                   target: (batch_size * sequence_lenght) A batch of sequences

               Outputs:
                   log_probs : (batch_size) Log likelihood for each example*
                   entropy: (batch_size) The entropies for the sequences. Not
                                         currently used.
        """
        len_con_token = 0
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token_dec = Variable(torch.zeros(batch_size_dec, 1).long())

        start_token[:] = self.voc.vocab['GO']

        # split train and target and con_token
        con_token = target[:, 0:len_con_token]

        x = torch.cat((start_token, target[:, len_con_token:-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, len_con_token:-1]), 1)


        y_target = target[:, len_con_token:].contiguous().view(-1)
        y_target_dec = target_decorator[:, len_con_token:].contiguous().view(-1)

        # 去掉token长度
        seq_length = seq_length - len(con_token[0])
        seq_length_dec = seq_length_dec - len(con_token[0])

        # log_probs = Variable(torch.zeros(batch_size))
        # entropy = Variable(torch.zeros(batch_size))

        logits = Prior_encode_forward(self.decodertf, x,x_dec)


        return logits

    def sample(self, batch_size, max_length=140, con_token_list= ['is_JNK3', 'is_GSK3', 'high_QED', 'good_SA']):
        """
               Sample a batch of sequences

               Args:
                   batch_size : Number of sequences to sample
                   max_length:  Maximum length of the sequences

               Outputs:
               seqs: (batch_size, seq_length) The sampled sequences.
               log_probs : (batch_size) Log likelihood for each sequence.
               entropy: (batch_size) The entropies for the sequences. Not
                                       currently used.
       """

        # conditional token
        con_token_list = Variable(self.voc.encode(con_token_list))

        con_tokens = Variable(torch.zeros(batch_size, len(con_token_list)).long())

        for ind, token in enumerate(con_token_list):
            con_tokens[:, ind] = token

        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        input_vector = start_token
        # print(batch_size)

        sequences = start_token
        log_probs = Variable(torch.zeros(batch_size))
        # log_probs1 = Variable(torch.zeros(batch_size))

        finished = torch.zeros(batch_size).byte()

        finished = finished.to(self.device)

        for step in range(max_length):
            logits = sample_forward_model(self.decodertf, input_vector, con_tokens)

            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)

            input_vector = torch.multinomial(prob, 1)

            # need to concat prior words as the sequences and input 记录下每一步采样
            sequences = torch.cat((sequences, input_vector), 1)


            log_probs += self._nll_loss(log_prob, input_vector.view(-1))
            # log_probs1 += NLLLoss(log_prob, input_vector.view(-1))
            # print(log_probs1==-log_probs)




            EOS_sampled = (input_vector.view(-1) == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)

            if torch.prod(finished) == 1:
                # print('End')
                break

            # because there are no hidden layer in transformer, so we need to append generated word in every step as the input_vector
            input_vector = sequences

        return sequences[:, 1:].data, log_probs

    def generate(self, batch_size, target, max_length=140):
        """
               Sample a batch of sequences

               Args:
                   batch_size : Number of sequences to sample
                   max_length:  Maximum length of the sequences

               Outputs:
               seqs: (batch_size, seq_length) The sampled sequences.
               log_probs : (batch_size) Log likelihood for each sequence.
               entropy: (batch_size) The entropies for the sequences. Not
                                       currently used.
       """
        len_con_token = 0
        # batch_size, seq_length = target.size()

        start_token = Variable(torch.zeros(batch_size, 1).long())

        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, len_con_token:-1]), 1)



        # con_tokens = Variable(torch.zeros(batch_size, len(con_token_list)).long())
        # print(con_token_list, 11111)
        #
        # for ind, token in enumerate(con_token_list):
        #     con_tokens[:,ind] = token

        input_vector = start_token
        # print(batch_size)

        sequences = start_token
        # log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        # entropy = Variable(torch.zeros(batch_size))

        finished = finished.to(self.device)

        for step in range(max_length):
            # print(step)
            # print(input_vector.size(), 123)
            logits = sample_forward_model(self.decodertf, input_vector,x)
            # print(logits.size(),222)
            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            # log_prob = F.log_softmax(logits_step, dim=1)

            input_vector = torch.multinomial(prob, 1)


            # need to concat prior words as the sequences and input 记录下每一步采样
            sequences = torch.cat((sequences, input_vector), 1)
            EOS_sampled = (input_vector.view(-1) == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            '''一次性计算所有step的nll'''
            if torch.prod(finished) == 1:
                # print('End')
                break

            # because there are no hidden layer in transformer, so we need to append generated word in every step as the input_vector
            input_vector = sequences

        return sequences[:, 1:].data

    def likelihood_rl(self, target,target_decorator):
        """
               Retrieves the likelihood of a given sequence

               Args:
                   target: (batch_size * sequence_lenght) A batch of sequences

               Outputs:
                   log_probs : (batch_size) Log likelihood for each example*
                   entropy: (batch_size) The entropies for the sequences. Not
                                         currently used.
        """
        len_con_token = 0
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        # start_token = Variable(torch.zeros(batch_size, 1).long())
        # start_token_dec = Variable(torch.zeros(batch_size_dec, 1).long())

        # start_token[:] = self.voc.vocab['GO']

        # split train and target and con_token
        # con_token = target[:, 0:len_con_token]

        # x = torch.cat((start_token, target[:, len_con_token:-1]), 1)
        # x_dec = torch.cat((start_token, target_decorator[:, len_con_token:-1]), 1)


        # y_target = target[:, len_con_token:].contiguous().view(-1)
        y_target_dec = target_decorator.contiguous().view(-1)

        tgt = torch.tensor(target)
        tgt_dec = torch.tensor(target_decorator)

        log_probs = Variable(torch.zeros(batch_size))
        entropy = Variable(torch.zeros(batch_size))

        logits = Prior_train_forward_rl(self.decodertf, tgt,tgt_dec)

        logits = F.log_softmax(logits,dim=2)
        # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        criterion = nn.NLLLoss(reduction="none", ignore_index=0)

        # criterion = F.nll_loss(output.view(-1, len(vocab)),
        #                   sm.contiguous().view(-1), ignore_index=PAD)

        # log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target)
        log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target_dec)
        print(logits.size(),123456789)

        # loss值
        mean_log_probs = log_probs.mean()

        # print('gooo',log_probs.shape)
        log_probs = log_probs.view(-1,  seq_length_dec)

        # 求和，得到每个分子的loss值
        log_probs_each_molecule = log_probs.sum(dim=1)
        return log_probs_each_molecule

    def likelihood_rl_reward(self, target,target_decorator,rl_loss,discounted_reward,gamma):
        """
               Retrieves the likelihood of a given sequence

               Args:
                   target: (batch_size * sequence_lenght) A batch of sequences

               Outputs:
                   log_probs : (batch_size) Log likelihood for each example*
                   entropy: (batch_size) The entropies for the sequences. Not
                                         currently used.
        """
        len_con_token = 0
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        # start_token_dec = Variable(torch.zeros(batch_size_dec, 1).long())

        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, len_con_token:-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, len_con_token:-1]), 1)


        # y_target = target[:, len_con_token:].contiguous().view(-1)
        y_target_dec = target_decorator.contiguous().view(-1)

        # tgt = torch.tensor(x)
        # tgt_dec = torch.tensor(x_dec)
        for step in range(seq_length_dec-1):
            logits = sample_forward_model(self.decodertf, x_dec[:,:step+1],x)
            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)
            top_i = x_dec[:,:step+2]
            rl_loss = rl_loss - (log_prob[0,top_i][0][-1].detach() * discounted_reward)
            discounted_reward = discounted_reward * gamma
            # log_probs =  NLLLoss(log_prob, x_dec[:,:step+1])


        return rl_loss,discounted_reward

    def generate_rl(self, batch_size, target, max_length=140):
        """
               Sample a batch of sequences

               Args:
                   batch_size : Number of sequences to sample
                   max_length:  Maximum length of the sequences

               Outputs:
               seqs: (batch_size, seq_length) The sampled sequences.
               log_probs : (batch_size) Log likelihood for each sequence.
               entropy: (batch_size) The entropies for the sequences. Not
                                       currently used.
       """
        len_con_token = 0
        # batch_size, seq_length = target.size()

        start_token = Variable(torch.zeros(batch_size, 1).long())

        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, len_con_token:-1]), 1)



        # con_tokens = Variable(torch.zeros(batch_size, len(con_token_list)).long())
        # print(con_token_list, 11111)
        #
        # for ind, token in enumerate(con_token_list):
        #     con_tokens[:,ind] = token

        input_vector = start_token
        # print(batch_size)

        sequences = start_token
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))

        finished = finished.to(self.device)

        for step in range(max_length):
            logits = sample_forward_model(self.decodertf, input_vector,x)
            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)

            input_vector = torch.multinomial(prob, 1)

            log_probs +=  NLLLoss(log_prob, input_vector)
            # print(log_probs.shape)
            entropy += -torch.sum((log_prob * prob), 1)

            # need to concat prior words as the sequences and input 记录下每一步采样
            sequences = torch.cat((sequences, input_vector), 1)
            EOS_sampled = (input_vector.view(-1) == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            '''一次性计算所有step的nll'''
            if torch.prod(finished) == 1:
                # print('End')
                break

            # because there are no hidden layer in transformer, so we need to append generated word in every step as the input_vector
            input_vector = sequences


        return sequences[:, 1:].data, log_probs, entropy



def Prior_train_forward(model, tgt, tgt_dec, con_token):
    tgt = torch.tensor(tgt)
    tgt_dec = torch.tensor(tgt_dec)

    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_dec_mask = gen_nopeek_mask(tgt_dec.shape[1])
    output = model(tgt,tgt_dec, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask,tgt_dec_key_padding_mask=None,tgt_dec_mask=tgt_dec_mask)

    del tgt, tgt_mask
    return output.squeeze(1)

def Prior_encode_forward(model, tgt, tgt_dec):
    tgt = torch.tensor(tgt)
    tgt_dec = torch.tensor(tgt_dec)

    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_dec_mask = gen_nopeek_mask(tgt_dec.shape[1])
    output = model.encode(tgt,tgt_dec, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask,tgt_dec_key_padding_mask=None,tgt_dec_mask=tgt_dec_mask)

    del tgt, tgt_mask
    return output

def Prior_train_forward_rl(model, tgt, tgt_dec):
    tgt = torch.tensor(tgt)
    tgt_dec = torch.tensor(tgt_dec)

    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_dec_mask = gen_nopeek_mask(tgt_dec.shape[1])
    output = model(tgt,tgt_dec, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask,tgt_dec_key_padding_mask=None,tgt_dec_mask=tgt_dec_mask)

    del tgt, tgt_mask
    return output.squeeze(1)


def sample_forward_model(model, tgt,tgt_scaffold):
    # src = torch.tensor(src).unsqueeze(0).long().to('cuda')
    # tgt = torch.tensor(tgt)
    tgt_scaffold = torch.tensor(tgt_scaffold)
    tgt_scaffold_mask = gen_nopeek_mask(tgt_scaffold .shape[1])

    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model(tgt_scaffold,tgt, tgt_key_padding_mask=None,
                   tgt_mask=tgt_scaffold_mask,tgt_dec_key_padding_mask=None,tgt_dec_mask=tgt_mask)
    # print(output.shape)

    return output

def sample_forward_model_scaffold(model, tgt):
    # src = torch.tensor(src).unsqueeze(0).long().to('cuda')
    # tgt = torch.tensor(tgt)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model(tgt, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask)
    # print(output.shape)

    return output


# 从一开始就不用遮住conditional token，mask从第len(conditonal)+len(start)之后开始
def gen_nopeek_mask(length):
    # mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask.to(device)

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).to(device)
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
