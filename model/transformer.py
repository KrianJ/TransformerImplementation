#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 15:43
# software: PyCharm-transformer
import torch
import torch.nn as nn

from layer.encoder_layer import EncoderLayer
from layer.decoder_layer import DecoderLayer
from model.encoder import TransformerEncoder
from model.transformerdecoder import TransformerDecoder
from component.embedding import TokenEmbedding, PositionalEncoding


class MyTransformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(MyTransformer, self).__init__()
        """ ----------编码器部分---------- """
        encoder_layer = EncoderLayer(d_model, n_head, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        """ ----------解码器部分---------- """
        decoder_layer = DecoderLayer(d_model, n_head, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()  # 初始化模型参数
        self.d_model = d_model
        self.n_head = n_head

    def _reset_parameters(self):
        """初始化模型参数"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @classmethod
    def generate_square_subsequent_mask(cls, sz):
        """
        生成注意力掩码矩阵
        :param sz: 序列长度
        :return:
        """
        # torch.triu: 返回二维矩阵, 包含输入矩阵 (2D 张量) 的上三角部分, 其余部分被设为 0
        # 将矩阵(sz, sz)上三角部分取False, 其他取True
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask中0的部分取-inf, 1的部分取0.
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # (sz,sz)

    def forward(self, src, tgt,
                src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Transformer的前向传播
        :param src: (src_len, batch_size, embed_dim)
        :param tgt: (tgt_len, batch_size, embed_dim)
        :param src_mask: (src_len, src_len), 一般为None, 输入编码不需要进行mask,本身就是要全部可见的
        :param tgt_mask: (tgt_len, tgt_len)
        :param memory_mask: (src_len, tgt_len)
        :param src_key_padding_mask: (batch_size, src_len)
        :param tgt_key_padding_mask: (batch_size, tgt_len)
        :param memory_key_padding_mask: (batch_size, src_len)
        :return:
        """
        memory = self.encoder(src=src,  # (src_len, batch_size, embed_dim)
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output  # (tgt_len, batch_size, embed_dim), embed_dim = n_head * k_dim


if __name__ == '__main__':
    d_model = 32
    n_head = 8
    n_enc_l = 6
    n_dec_l = 6
    dim_ff = 500
    dropout = 0.1
    src_len = 5
    tgt_len = 6
    batch_size = 2

    src = torch.rand((src_len, batch_size, d_model))
    src_key_padding_mask = torch.tensor(
        [[True, True, True, False, False],
         [True, True, True, True, False]]
    )

    tgt = torch.rand((tgt_len, batch_size, d_model))  # shape: [tgt_len, batch_size, embed_dim]
    tgt_key_padding_mask = torch.tensor(
        [[True, True, True, False, False, False],
         [True, True, True, True, False, False]]
    )

    model = MyTransformer(d_model, n_head, n_enc_l, n_dec_l, dim_ff, dropout)
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    output = model(src=src, tgt=tgt, tgt_mask=tgt_mask,
                   src_key_padding_mask=src_key_padding_mask,
                   tgt_key_padding_mask=tgt_key_padding_mask,
                   memory_key_padding_mask=src_key_padding_mask)
    print(output.shape)
