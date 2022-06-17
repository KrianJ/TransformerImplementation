#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 15:07
# software: PyCharm-decoder_layer
import torch.nn as nn
import torch.nn.functional as F

from component.multi_head_attention import MyMultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        """

        :param d_model: d_k = d_v = d_model / n_head = 64, 模型中向量的维度，论文默认值为 512
        :param n_head: 多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout: 丢弃率，论文中的默认值为 0.1
        """
        super(DecoderLayer, self).__init__()
        # 解码部分输入的多头注意力, 论文中的masked multi-head attention
        self.self_attn = MyMultiHeadAttention(embed_dim=d_model, n_heads=n_head, dropout=dropout)
        # 交互部分的多头注意力(memory + 解码部分的输入编码)
        self.multi_head_attn = MyMultiHeadAttention(embed_dim=d_model, n_heads=n_head, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Decoder部分的3个层归一化(Norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param tgt: (tgt_len, batch_size, embed_dim), DecoderLayer的输入序列target
        :param memory: (src_len, batch_size, embed_dim), TransformerEncoder的输出(memory)
        :param tgt_mask: (tgt_len, tgt_len), 注意力mask输入,用于掩盖当前position之后的信息
        :param memory_mask: (src_len, src_len), 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: (batch_size, tgt_len), DecoderLayer输入的padding情况
        :param memory_key_padding_mask: (batch_size, src_len), 即src_key_padding_mask, TransformerEncoder输入的padding情况
        :return:
        """
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        tgt2 = self.self_attn(tgt, tgt, tgt,        # (tgt_len, batch_size, embed_dim)
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # Add&Norm-1 残差连接
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)           # (tgt_len, batch_size, embed_dim)

        # 与memory进行交互,计算多头注意力
        tgt2 = self.multi_head_attn(tgt, memory, memory,
                                    attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        # Add&Norm-2
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)           # (tgt_len, batch_size, embed_dim)

        # FeedForward, 两层全连接
        tgt2 = self.linear1(tgt)        # (tgt_len, batch_size, dim_feedforward)
        tgt2 = self.dropout(self.activation(tgt2))
        tgt2 = self.linear2(tgt2)       # (tgt_len, batch_size, embed_dim)

        # Add&Norm-3
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)           # (tgt_len, batch_size, embed_dim), embed_dim = n_head * k_dim
        return tgt



