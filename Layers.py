''' Define the Layers '''
import torch.nn as nn
from SubLayers import MultiHeadAttention, PositionWiseFeedForward

class EncoderLayer(nn.Module):

    ''' composed of two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k,d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn
