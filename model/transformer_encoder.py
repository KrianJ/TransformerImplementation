#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 12:18
# software: PyCharm-encoder

import torch.nn as nn
from toolkit.utils import get_clones


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        :param encoder_layer: 单个编码层
        :param num_layers: 编码层数量, 论文默认6
        :param norm: 归一化层
        """
        super(TransformerEncoder, self).__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        参数同layer.encoder_layer.EncoderLayer.forward
        :param src:
        :param mask:
        :param src_key_padding_mask:
        :return:
        """
        output = src
        # 多个encoder堆叠前向传播
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output  # (src_len, batch_size, num_heads * kdim) <==> (src_len,batch_size,embed_dim)
