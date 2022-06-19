#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 15:33
# software: PyCharm-decoder
import copy

import torch
import torch.nn as nn
from toolkit.utils import get_clones


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        解码器
        :param decoder_layer: 单个DecoderLayer
        :param num_layers: 解码层数量, 论文默认6
        :param norm: 归一化
        """
        super(TransformerDecoder, self).__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt        # (tgt_len, batch_size, embed_dim)
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output       # (tgt_len, batch_size, embed_dim)
