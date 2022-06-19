#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/18 17:51
# software: PyCharm-bert_embedding
""" Bert的embedding部分, 包括三个部分:
Token Embedding: 普通的token id对应的embedding
Positional Embedding: 位置编码, 与transformer不同, transformer是计算的固定位置编码, bert是初始化的普通embedding,参与后续训练进行更新.
Segment Embedding: 段落编码, 满足成对文本输入的场景,用于区分两个文本序列。
"""
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        """
        Token Embedding
        :param vocab_size:
        :param hidden_size:
        :param pad_token_id:
        :param initializer_range:
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._reset_parameter(initializer_range)

    def _reset_parameter(self, initializer_range):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0., std=initializer_range)

    def forward(self, input_ids):
        """
        :param input_ids: shape : [input_ids_len, batch_size]
        :return: shape: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        """
        位置编码。
        *** 注意： Bert中的位置编码完全不同于Transformer中的位置编码，
        前者本质上也是一个普通的Embedding层，而后者是通过公式计算得到，
        而这也是为什么Bert只能接受长度为512字符的原因，因为位置编码的最大size为512 ***
        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
                                                      ————————  GoogleResearch
        https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py

        """
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self._reset_parameter(initializer_range)

    def _reset_parameter(self, initializer_range):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0., std=initializer_range)

    def forward(self, position_ids):
        """
        :param position_ids: [1,position_ids_len]
        :return: [position_ids_len, 1, hidden_size]
        """
        return self.embedding(position_ids).transpose(0, 1)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        """
        :param token_type_ids:  shape: [token_type_ids_len, batch_size]
        :return: shape: [token_type_ids_len, batch_size, hidden_size]
        """
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=initializer_range)


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        pass