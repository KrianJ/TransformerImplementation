#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 11:54
# software: PyCharm-encoder

import torch.nn as nn
import torch.nn.functional as F

from component.multi_head_attention import MyMultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        """
        Transformer的encoderLayer
        :param d_model: d_k = d_v = d_model / n_head = 64, 模型的向量维度,论文默认512
        :param n_head: 多头注意力个数
        :param dim_feedforward: 全连接向量中的维度, 论文默认2048
        :param dropout: dropout, 论文默认0.1
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MyMultiHeadAttention(d_model, n_head, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分输入, (src_len, batch_size, embed_dim)
        :param src_mask: src掩码, (batch_size, src_len)
        :param src_key_padding_mask: src的padding情况
        :return:
        """
        # 计算多头注意力 -> src2: (src_len, batch_size, n_head * k_dim), n_head * k_dim = embed_dim
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # 多头注意力之后的add_norm部分
        src = src + self.dropout1(src2)     # Add(残差连接)
        src = self.norm1(src)               # Norm
        # FeedForward部分
        src2 = self.activation(self.linear1(src))   # (src_len, batch_size, dim_feedforward)
        src2 = self.linear2(self.dropout(src2))     # (src_len, batch_size, n_head * k_dim)
        # FeedForward之后的add_norm部分
        src = src + self.dropout2(src2)     # Add
        src = self.norm2(src)               # Norm

        return src      # (src_len, batch_size, num_heads * k_dim) <==> (src_len,batch_size,embed_dim)




