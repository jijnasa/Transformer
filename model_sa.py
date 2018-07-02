u''' Transformer Model '''
import torch 
import torch.nn as nn
import numpy as np
import Constants as Constants
from Layers import EncoderLayer
from torch.autograd import Variable
from torch import LongTensor
#import tensorflow as tf
pos = Variable(torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]))
def get_attn_padding_mask(seq_q, seq_k):
    '''Indicate Passing-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1) #bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) #bxsqxsk
    return pad_attn_mask

def position_encoding_init(n_position, d_pos_vec):
    '''Init the sinusoid position encoding table '''

    #keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)] 
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class Encoder(nn.Module):
    ''' An encoder model with self attention mechanism. '''

    def __init__(self, word_emb, rela_emb, max_len, n_layers=6, n_head=8, d_k=64, d_v=64,d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        super(Encoder, self).__init__()

        n_position = max_len + 1
        self.max_len = max_len
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        #Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.eord_embedding_weightn = nn.Parameter(torch.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix

        #Rela Embedding Layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
        self.rela_embedding.weight = nn.Parameter(torch.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = False # fix the embedding matrix

        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)for _ in range(n_layers)])
    def forward(self, q, pos, return_attns=False):
        #Word embedding look up
        enc_input = q


        #Position Encoding addition
        n = self.position_enc(pos)
        #print('hello')

        enc_input += self.position_enc(pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(ques_rel, ques_rel)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism '''
    def __init__(self, word_emb, rela_emb, q_len, r_len,n_layers=6, n_head=8, d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64, dropout=0.1):
        
        super(Transformer, self).__init__()
        self.encoder_question = Encoder(word_emb, rela_emb, q_len, n_layers=n_layers, n_head=n_head, d_word_vec=d_word_vec, d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.encoder_rela = Encoder(word_emb, rela_emb, r_len, n_layers=n_layers, n_head=n_head, d_word_vec=d_word_vec, d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
    def get_trainable_parameters(self):
        '''Avoid updating the position encoding '''
        enc_freezed_param_ids_ques = set(map(id, self.encoder_question.position_enc.parameters()))
        enc_freezed_param_ids_rela = set(map(id, self.encoder_rela.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids_ques | enc_freezed_param_ids_rela
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)
    def forward(self, q, rela_relation, word_relation):
        #q = torch.transpose(q, 0, 1)
        #print(len(q))
        #print(len(q[0])
        #input('Enter')
        print(q)
        print(type(q))
        s = Variable(LongTensor(q))
        print(s.size())
        #q = q.cuda()
        #print(type(q))
        global pos
        a = q[0:1]
        #pos = Variable(torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]))
        pos = pos.view(1,35)
        pos[a==0] = 0
        for i in range(2,409):
            a = q[i-1:i]
            #print('one row')
            #print(a)
            b = a.clone()
            c = Variable(torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]))
            c = c.view(1,35)
            #print(c)
            #print(Variable(LongTensor(c)).size())
            #print(Variable(LongTensor(b)).size())
            c[b==0] = 0
            #print('Resul:')
            #print(c)
            #global pos
            pos = torch.cat((pos,c),0)
            #print('After concatenation')
            #print(pos)
       # prin:t(a[0])
        q = self.word_embedding(q)
        q = self.dropout(q)
        print(pos)
        v = Variable(LongTensor(pos)).size()
        print(v)
        rela_relation = torch.transpose(rela_relation, 0, 1)
        rela_relation = self.rela_embedding(rela_relation)
        word_relation = torch.transpose(word_relation, 0, 1)
        word_relation = self.word_embedding(word_relation)
        r = torch.cat([rela_relation, word_relation], 0)
        #ques_rel[:, :-1], pos[:, :-1] = q
        #ques_rel = ques_rel[:, :-1]
        #pos = pos[:, :-1]
        #pos = [i for i, x in enumerate(q)]
       # print(pos)
        enc_output_ques, *_ = self.encoder_question(q,pos)

        ques_rel, pos = r
        encoder_output_rela, *_ = self.encoder_rela(ques_rel, pos)

        score = self.cos(encoder_output_ques, encoder_output_rela)
        return score
