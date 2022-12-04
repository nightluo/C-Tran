import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
from .utils import get_activation_fn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        # 经典的返回注意力输出结果，其他与 nn.TransformerEncoderLayer 一致
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask= None, src_key_padding_mask = None):
        # Multi-Head Attention, attn_output, attn_output_weights = self.self_attn(query, key, value)
        # query = src * W_Q
        # key = src * W_K
        # value = src * W_V
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # Add, 先 dropout 再 residual connect
        src = src + self.dropout1(src2) 
        # LayerNorm
        src = self.norm1(src)
        # Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add
        src = src + self.dropout2(src2)
        # LayerNorm
        src = self.norm2(src)

        # 返回 Encoder 结果和 attn 参数矩阵
        return src, attn
        

class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead = 4,dropout=0.1):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model, nhead, d_model * 1, dropout=dropout, activation='relu')
        # why ? not nn.TransformerEncoderLayer
        # nn.TransformerEncoderLayer 不返回 attn 参数矩阵
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu') 

    def forward(self, k, mask=None):
        attn = None
        k = k.transpose(0, 1)  
        x, attn = self.transformer_layer(k, src_mask=mask)
        # x = self.transformer_layer(k,src_mask=mask)
        x=x.transpose(0, 1)
        return x, attn