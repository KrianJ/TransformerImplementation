#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 10:53
# software: PyCharm-embedding
""" Transformer的embedding实现 """
import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, tokens):
        """
        将token转换成embedding
        :param tokens: (len, batch_size)
        :return: (len, batch_size, embed_size)
        """
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)


class PositionalEncoding(nn.Module):
    """
    每个位置编码的计算公式为
    PE_(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """

        :param d_model:
        :param dropout:
        :param max_len: 序列最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)                                      # (max_len, d_model), 初始化位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (max_len, 1), 位置索引
        # 位置索引的分母部分, 将1 / 10000^(2i/d_model)化简得到exp(2i * (-log(10000) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2, )
        pe[:, 0::2] = torch.sin(position * div_term)    # (max_len, d_model / 2), 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)    # (max_len, d_model / 2), 奇数位置
        pe = pe.unsqueeze(0).transpose(0, 1)            # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        对x进行位置编码
        :param x: (x_len, batch_size, embed_size), token embedding
        :return: (x_len, batch_size, embed_size)
        """
        # 在位置矩阵中取与输入序列长度相等的前x_len行，再加上原始的embedding
        x = x + self.pe[:x.size(0), :]      # (x_len, batch_size, d_model)
        return self.dropout(x)


if __name__ == '__main__':
    x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    x = x.reshape(5, 2)  # [src_len, batch_size]
    token_embedding = TokenEmbedding(vocab_size=11, embed_size=512)
    x = token_embedding(tokens=x)
    pos_embedding = PositionalEncoding(d_model=512)
    x = pos_embedding(x=x)
    print(x.shape)  # torch.Size([5, 2, 512])
