#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as tnnur
from utils.utils import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


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


class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))

# class RNN():
#     """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
#     order to determine size of the vocabulary and index of the END token"""
#     def __init__(self, voc):
#         self.rnn_encoder = MultiGRU(voc.vocab_size)
#         self.rnn_decoder =MultiGRU(voc.vocab_size)
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#             self.rnn_encoder.to(self.device)
#             self.rnn_decoder.to(self.device)
#         self.voc = voc
#
#     def likelihood(self, target,target_decorator):
#         """
#             Retrieves the likelihood of a given sequence
#
#             Args:
#                 target: (batch_size * sequence_lenght) A batch of sequences
#
#             Outputs:
#                 log_probs : (batch_size) Log likelihood for each example*
#                 entropy: (batch_size) The entropies for the sequences. Not
#                                       currently used.
#         """
#         batch_size, seq_length = target.size()
#         batch_size_dec, seq_length_dec = target_decorator.size()
#
#
#         start_token = Variable(torch.zeros(batch_size, 1).long())
#         start_token[:] = self.voc.vocab['GO']
#
#         x = torch.cat((start_token, target[:, :-1]), 1)
#         x_dec = torch.cat((start_token, target_decorator[:, :-1]), 1)
#
#         h = self.rnn_encoder.init_h(batch_size)
#
#         log_probs = Variable(torch.zeros(batch_size))
#         # entropy = Variable(torch.zeros(batch_size))
#         for step in range(seq_length):
#             logits, h = self.rnn_encoder(x[:, step], h)
#
#         for step in range(seq_length_dec):
#             logits, h = self.rnn_decoder(x_dec[:, step], h)
#             log_prob = F.log_softmax(logits)
#             prob = F.softmax(logits)
#             # A = target[:, step]
#             log_probs += NLLLoss(log_prob, target_decorator[:, step])
#             # entropy += -torch.sum((log_prob * prob), 1)
#         return log_probs
#
#     def sample(self, batch_size, max_length=140):
#         """
#             Sample a batch of sequences
#
#             Args:
#                 batch_size : Number of sequences to sample
#                 max_length:  Maximum length of the sequences
#
#             Outputs:
#             seqs: (batch_size, seq_length) The sampled sequences.
#             log_probs : (batch_size) Log likelihood for each sequence.
#             entropy: (batch_size) The entropies for the sequences. Not
#                                     currently used.
#         """
#         start_token = Variable(torch.zeros(batch_size).long())
#         start_token[:] = self.voc.vocab['GO']
#         h = self.rnn.init_h(batch_size)
#         x = start_token
#
#
#         sequences = []
#         log_probs = Variable(torch.zeros(batch_size))
#         print('batch is:',batch_size)
#
#         finished = torch.zeros(batch_size).byte()
#         entropy = Variable(torch.zeros(batch_size))
#         if torch.cuda.is_available():
#             finished = finished.to(device)
#
#         for step in range(max_length):
#             logits, h = self.rnn(x, h)
#             prob = F.softmax(logits)
#             log_prob = F.log_softmax(logits)
#             x = torch.multinomial(prob, 1).view(-1)
#             sequences.append(x.view(-1, 1))
#             log_probs +=  NLLLoss(log_prob, x)
#             # print(log_probs.shape)
#             entropy += -torch.sum((log_prob * prob), 1)
#
#             x = Variable(x.data)
#             EOS_sampled = (x == self.voc.vocab['EOS']).data
#             finished = torch.ge(finished + EOS_sampled, 1)
#             if torch.prod(finished) == 1:
#                 break
#         # a = sequences[1:3]
#         # b = torch.cat(a, 1)
#         sequences = torch.cat(sequences, 1)
#         return sequences.data, log_probs, entropy

class Encoder(nn.Module):
    """
    Simple bidirectional RNN encoder implementation.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = nn.Sequential(
            nn.Embedding(self.vocabulary_size, self.num_dimensions),
            nn.Dropout(dropout)
        )
        self._rnn = nn.GRU(self.num_dimensions,int(self.num_dimensions) , self.num_layers,
                             batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, padded_seqs, seq_lengths):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the sequences (batch, seq).
        :param seq_lengths: The lengths of the sequences (for packed sequences).
        :return : A tensor with all the output values for each step and the two hidden states.
        """
        batch_size = padded_seqs.size()[0]
        max_seq_size = padded_seqs.size()[1]
        seq_lengths=torch.tensor([seq_lengths])
        hidden_state = self._initialize_hidden_state(batch_size)

        padded_seqs = self._embedding(padded_seqs)
        hs_h, hs_c = (hidden_state, hidden_state.clone().detach())
        # packed_seqs = tnnur.pack_padded_sequence(padded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_seqs,hs_h = self._rnn(padded_seqs,hs_h)
        # padded_seqs, _ = tnnur.pad_packed_sequence(packed_seqs, batch_first=True)

        # sum up bidirectional layers and collapse
        hs_h = hs_h.view(self.num_layers, 2, batch_size, self.num_dimensions)\
            .sum(dim=1).squeeze()  # (layers, batch, dim)
        hs_c = hs_c.view(self.num_layers, 2, batch_size, self.num_dimensions)\
            .sum(dim=1).squeeze()  # (layers, batch, dim)
        padded_seqs = packed_seqs.view(batch_size, max_seq_size, 2, self.num_dimensions)\
            .sum(dim=2).squeeze()  # (batch, seq, dim)
        # print(packed_seqs.size(),3333)

        return padded_seqs, hs_h#, hs_c)

    def _initialize_hidden_state(self, batch_size):
        if torch.cuda.is_available():
         return torch.zeros(self.num_layers*2, batch_size, self.num_dimensions).cuda()
        else:
         return torch.zeros(self.num_layers * 2, batch_size, self.num_dimensions)

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "num_layers": self.num_layers,
            "num_dimensions": self.num_dimensions,
            "vocabulary_size": self.vocabulary_size,
            "dropout": self.dropout
        }

class Decoder(nn.Module):
    """
    Simple RNN decoder.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = nn.Sequential(
            nn.Embedding(self.vocabulary_size, self.num_dimensions),
            nn.Dropout(dropout)
        )
        self._rnn = nn.GRU(self.num_dimensions, self.num_dimensions, self.num_layers,
                             batch_first=True, dropout=self.dropout, bidirectional=False)

        self._attention = AttentionLayer(self.num_dimensions)

        self._linear = nn.Linear(self.num_dimensions, self.vocabulary_size)  # just to redimension

    def forward(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param seq_lengths: A list with the length of each output sequence.
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param hidden_states: The hidden states from the encoder.
        :return : Three tensors: The output logits, the hidden states of the decoder and the attention weights.
        """
        padded_encoded_seqs = self._embedding(padded_seqs)
        # seq_lengths=torch.tensor([seq_lengths])
        # packed_encoded_seqs = tnnur.pack_padded_sequence(
        #     padded_encoded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        # print(hidden_states.size())
        padded_encoded_seqs, hidden_states = self._rnn(padded_encoded_seqs, hidden_states)
        # padded_encoded_seqs, _ = tnnur.pad_packed_sequence(packed_encoded_seqs, batch_first=True)  # (batch, seq, dim)

        mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(padded_encoded_seqs, encoder_padded_seqs, mask)
        logits = self._linear(attn_padded_encoded_seqs)*mask  # (batch, seq, voc_size)
        return logits, hidden_states, attention_weights

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "num_layers": self.num_layers,
            "num_dimensions": self.num_dimensions,
            "vocabulary_size": self.vocabulary_size,
            "dropout": self.dropout
        }

class Decorator(nn.Module):
    """
    An encoder-decoder that decorates scaffolds.
    """

    # def __init__(self, encoder_params, decoder_params):
    #     super(Decorator, self).__init__()
    #
    #     self._encoder = Encoder(**encoder_params)
    #     self._decoder = Decoder(**decoder_params)

    def __init__(self, num_layers,num_dimensions,vocab_size,dropout):
        super(Decorator, self).__init__()
        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocab_size = vocab_size
        self.dropout = dropout
        self._encoder = Encoder(self.num_layers,self.num_dimensions,vocab_size,self.dropout)
        self._decoder = Decoder(self.num_layers,self.num_dimensions,vocab_size,self.dropout)

    def forward(self, encoder_seqs, encoder_seq_lengths, decoder_seqs, decoder_seq_lengths):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param encoder_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_seq_lengths: A list with the length of each input sequence.
        :param decoder_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_seq_lengths: The lengths of the decoder sequences.
        :return : The output logits as a tensor (batch, seq_d, dim).
        """
        encoder_padded_seqs, hidden_states = self.forward_encoder(encoder_seqs, encoder_seq_lengths)
        logits, _, attention_weights = self.forward_decoder(
            decoder_seqs, decoder_seq_lengths, encoder_padded_seqs, hidden_states)
        return logits, attention_weights

    def forward_encoder(self, padded_seqs, seq_lengths):
        """
        Does a forward pass only of the encoder.
        :param padded_seqs: The data to feed the encoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns a tuple with (encoded_seqs, hidden_states)
        """
        return self._encoder(padded_seqs, seq_lengths)

    def forward_decoder(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):
        """
        Does a forward pass only of the decoder.
        :param hidden_states: The hidden states from the encoder.
        :param padded_seqs: The data to feed to the decoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns the logits, the hidden state for each element of the sequence passed and the attention weights.
        """
        return self._decoder(padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states)

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "encoder_params": self._encoder.get_params(),
            "decoder_params": self._decoder.get_params()
        }

class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self,num_layers, num_dimensions, voc,dropout):
        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.voc = voc
        self.dropout = dropout
        self.decorator =Decorator(self.num_layers,self.num_dimensions,voc.vocab_size,self.dropout)
        # self.rnn_encoder = Encoder(self.num_layers,self.num_dimensions,voc.vocab_size,self.dropout)
        # self.rnn_decoder =Decoder(self.num_layers,self.num_dimensions,voc.vocab_size,self.dropout)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decorator.to(self.device)
            # self.rnn_encoder.to(self.device)
            # self.rnn_decoder.to(self.device)


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
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, :-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, :-1]), 1)
        y_target_dec = target_decorator.contiguous().view(-1)

        tgt = torch.tensor(x)
        tgt_dec = torch.tensor(x_dec)

        logits, attention_weights=self.decorator(tgt, seq_length,tgt_dec, seq_length_dec)
        # encoder_padded_seqs, hidden_states = self.rnn_encoder(tgt, seq_length)
        # logits, _, attention_weights = self.rnn_decoder(
        #     tgt_dec, seq_length_dec, encoder_padded_seqs, hidden_states)

        logits = logits.log_softmax(dim=2)#.transpose(1, 2)  # (batch, voc, seq - 1)
        # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        criterion = nn.NLLLoss(reduction="none", ignore_index=0)


        log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target_dec)

        # loss值
        mean_log_probs = log_probs.mean()

        # print('gooo',log_probs.shape)
        log_probs = log_probs.view(-1,  seq_length_dec)

        # 求和，得到每个分子的loss值
        log_probs_each_molecule = log_probs.sum(dim=1)

        return mean_log_probs, log_probs_each_molecule

        # logits = self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)  # (batch)

    def likelihood_2(self, target,target_decorator):  #wrong
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, :-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, :-1]), 1)
        # y_target_dec = target_decorator.contiguous().view(-1)

        tgt = torch.tensor(x)
        tgt_dec = torch.tensor(x_dec)

        log_probs = Variable(torch.zeros(batch_size))

        encoder_padded_seqs, hidden_states = self.decorator.forward_encoder(tgt, seq_length)

        for step in range(seq_length_dec):
            # logits, h = self.rnn(x, h)
            logits, hidden_states,_ = self.decorator.forward_decoder(
                x_dec[:,step], seq_length_dec, encoder_padded_seqs, hidden_states)

            prob = F.softmax(logits,dim=1)
            log_prob = F.log_softmax(logits,dim=1)
            log_probs +=  NLLLoss(log_prob, target_decorator[:, step])
        mean_log_probs = log_probs.mean()


        return mean_log_probs,log_probs

        # logits = self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)  # (batch)  #wrong

    def sample(self, batch_size,target, max_length=140):
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
        batch_size, seq_length = target.size()

        start_token = Variable(torch.zeros(batch_size,1).long())
        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, :-1]), 1)
        tgt = torch.tensor(x)

        encoder_padded_seqs, hidden_states = self.decorator.forward_encoder(tgt, seq_length)
        # h = self.rnn.init_h(batch_size)
        # x = start_token
        input_vector = start_token
        seq_lengths = torch.ones(batch_size)
        sequences = start_token
        log_probs = Variable(torch.zeros(batch_size))
        print('batch is:',batch_size)

        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))

        if torch.cuda.is_available():
            finished = finished.to(device)

        for step in range(max_length):
            # logits, h = self.rnn(x, h)
            logits, hidden_states,_ = self.decorator.forward_decoder(
                input_vector, seq_lengths, encoder_padded_seqs, hidden_states)
            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step,dim=1)
            log_prob = F.log_softmax(logits_step,dim=1)
            input_vector = torch.multinomial(prob, 1)


            log_probs +=  NLLLoss(log_prob, input_vector)
            # print(log_probs.shape)
            entropy += -torch.sum((log_prob * prob), 1)

            sequences = torch.cat((sequences, input_vector), 1)
            EOS_sampled = (input_vector == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break
            input_vector = sequences
        # a = sequences[1:3]
        # b = torch.cat(a, 1)
        # sequences = torch.cat(sequences, 1)
        return sequences[:, 1:].data
        # return sequences.data, log_probs, entropy

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
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, :-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, :-1]), 1)
        y_target_dec = target_decorator.contiguous().view(-1)

        tgt = torch.tensor(x)
        tgt_dec = torch.tensor(x_dec)

        logits, attention_weights=self.decorator(tgt, seq_length,tgt_dec, seq_length_dec)


        logits = logits.log_softmax(dim=2)#.transpose(1, 2)  # (batch, voc, seq - 1)
        # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        criterion = nn.NLLLoss(reduction="none", ignore_index=0)


        log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target_dec)
        print(log_probs.size())
        # logits = self.NLLLoss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)

        # loss值
        mean_log_probs = log_probs.mean()

        # print('gooo',log_probs.shape)
        log_probs = log_probs.view(-1,  seq_length_dec)
        print(log_probs.size())

        # 求和，得到每个分子的loss值
        log_probs_each_molecule = log_probs.sum(dim=1)
        # log_probs_each_molecule = log_probs.mean(dim=1)

        return log_probs_each_molecule
        # return mean_log_probs

        # logits = self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)  # (batch)

    def likelihood_rl_reward(self, target,target_decorator):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        batch_size_dec, seq_length_dec = target_decorator.size()


        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, :-1]), 1)
        x_dec = torch.cat((start_token, target_decorator[:, :-1]), 1)
        y_target_dec = target_decorator.contiguous().view(-1)

        tgt = torch.tensor(x)
        tgt_dec = torch.tensor(x_dec)

        encoder_padded_seqs, hidden_states = self.decorator.forward_encoder(tgt, seq_length)
        # h = self.rnn.init_h(batch_size)
        # x = start_token
        seq_lengths = torch.ones(batch_size)
        log_probs = Variable(torch.zeros(batch_size))

        for step in range(seq_length_dec-1):
            # logits, h = self.rnn(x, h)
            logits, hidden_states,_ = self.decorator.forward_decoder(
                tgt_dec[:,step+1].unsqueeze(dim=1), seq_lengths, encoder_padded_seqs, hidden_states)
            # logits_step = logits[:, step, :]
            logits = logits.squeeze(dim=1)
            prob = F.softmax(logits,dim=1)
            log_prob = F.log_softmax(logits,dim=1)
            log_probs += NLLLoss(log_prob, target_decorator[:,step+1])
            # input_vector = torch.multinomial(prob, 1)

        #     log_probs += -NLLLoss(log_prob, input_vector) #加了负号
        #     # print(log_probs.shape)
        #     entropy += -torch.sum((log_prob * prob), 1)
        #
        #     sequences = torch.cat((sequences, input_vector), 1)
        #     EOS_sampled = (input_vector == self.voc.vocab['EOS']).data
        #     finished = torch.ge(finished + EOS_sampled, 1)
        #     if torch.prod(finished) == 1:
        #         break
        #     input_vector = sequences
        # logits = logits.log_softmax(dim=2)#.transpose(1, 2)  # (batch, voc, seq - 1)
        # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        # # criterion = nn.NLLLoss(reduction="none", ignore_index=0)
        #
        #
        # log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target_dec)
        #
        # # loss值
        # mean_log_probs = log_probs.mean()
        #
        # # print('gooo',log_probs.shape)
        # log_probs = log_probs.view(-1,  seq_length_dec)
        #
        #
        # # 求和，得到每个分子的loss值
        # log_probs_each_molecule = log_probs.sum(dim=1)
        return log_probs
        # return mean_log_probs

        # logits = self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)  # (batch)

    # def likelihood_rl_reward(self, target,target_decorator,rl_loss,discounted_reward,gamma):
    #     """
    #         Retrieves the likelihood of a given sequence
    #
    #         Args:
    #             target: (batch_size * sequence_lenght) A batch of sequences
    #
    #         Outputs:
    #             log_probs : (batch_size) Log likelihood for each example*
    #             entropy: (batch_size) The entropies for the sequences. Not
    #                                   currently used.
    #     """
    #     batch_size, seq_length = target.size()
    #     batch_size_dec, seq_length_dec = target_decorator.size()
    #
    #     start_token = Variable(torch.zeros(batch_size, 1).long())
    #     start_token[:] = self.voc.vocab['GO']
    #
    #     x = torch.cat((start_token, target[:, :-1]), 1)
    #     x_dec = torch.cat((start_token, target_decorator[:, :-1]), 1)
    #     y_target_dec = target_decorator.contiguous().view(-1)
    #
    #     tgt = torch.tensor(x)
    #     tgt_dec = torch.tensor(x_dec)
    #
    #     logits, attention_weights = self.decorator(tgt, seq_length, tgt_dec, seq_length_dec)
    #
    #     logits = logits.log_softmax(dim=2)  # .transpose(1, 2)  # (batch, voc, seq - 1)
    #     criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    #     # criterion = nn.NLLLoss(reduction="none", ignore_index=0)
    #
    #     log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target_dec)
    #
    #     # loss值
    #     mean_log_probs = log_probs.mean()
    #
    #     log_probs = log_probs.view(-1,  seq_length_dec)
    #
    #
    #     # 求和，得到每个分子的loss值
    #     log_probs_each_molecule = log_probs.sum(dim=1)
    #     return rl_loss,discounted_reward
    #     # return mean_log_probs
    #
    #     # logits = self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)  # (batch)

    def sample_rl(self, batch_size,target, max_length=140):
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
        batch_size, seq_length = target.size()

        start_token = Variable(torch.zeros(batch_size,1).long())
        start_token[:] = self.voc.vocab['GO']

        x = torch.cat((start_token, target[:, :-1]), 1)
        tgt = torch.tensor(x)

        encoder_padded_seqs, hidden_states = self.decorator.forward_encoder(tgt, seq_length)
        # h = self.rnn.init_h(batch_size)
        # x = start_token
        input_vector = start_token
        seq_lengths = torch.ones(batch_size)
        sequences = start_token
        log_probs = Variable(torch.zeros(batch_size))
        # print('batch is:',batch_size)

        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))


        if torch.cuda.is_available():
            finished = finished.to(device)
        # hidden_states=hidden_states.unsqueeze(dim=1)
        # encoder_padded_seqs = encoder_padded_seqs.unsqueeze(dim=0)


        for step in range(max_length):
            # logits, h = self.rnn(x, h)
            logits, hidden_states,_ = self.decorator.forward_decoder(
                input_vector, seq_lengths, encoder_padded_seqs, hidden_states)
            # logits_step = logits[:, step, :]

            prob = logits.softmax(dim=2).squeeze()
            log_prob = logits.log_softmax(dim=2).squeeze()
            input_vector = torch.multinomial(prob, 1)

            log_probs += NLLLoss(log_prob, input_vector)
            # print(log_probs.shape)
            entropy += -torch.sum((log_prob * prob), 1)

            sequences = torch.cat((sequences, input_vector), 1)
            EOS_sampled = (input_vector == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break
            # input_vector = sequences
        # a = sequences[1:3]
        # b = torch.cat(a, 1)
        # sequences = torch.cat(sequences, 1)
        # return sequences[:, 1:].data
        return sequences[:, 1:].data, log_probs, entropy
        # return sequences.data, log_probs, entropy

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
