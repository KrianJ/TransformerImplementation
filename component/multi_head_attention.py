#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/15 21:35
# software: PyCharm-multi_head_attention

"""多头注意力实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def multi_head_attention_forward(
        query,      # (tgt_len, batch_size, embed_dim)
        key,        # (src_len, batch_size, embed_dim)
        value,      # (src_len, batch_size, embed_dim)
        num_heads,
        dropout_p,
        out_proj_weight,        # (embed_dim, embed_dim) embed_dim = v_dim * num_heads
        out_proj_bias,
        training=True,
        key_padding_mask=None,  # (batch_size, src_len/tgt_len)
        q_proj_weight=None,     # (embed_dim, k_dim * num_heads)
        k_proj_weight=None,     # (embed_dim, k_dim * num_heads)
        v_proj_weight=None,     # (embed_dim, v_dim * num_heads)
        attn_mask=None          # (tgt_len,src_len)
):
    """ 第一阶段, 计算得到Q, K, V """
    q = F.linear(query, q_proj_weight)  # -> (tgt_len, batch_size, k_dim * num_heads)
    k = F.linear(key, k_proj_weight)    # -> (src_len, batch_size, k_dim * num_heads)
    v = F.linear(value, v_proj_weight)  # -> (src_len, batch_size, v_dim * num_heads)

    """ 第二阶段, 缩放以及attn_mask维度判断 """
    tgt_len, bsz, embed_dim = query.size()  # (tgt_len, batch_size, embed_dim)
    src_len = key.size(0)
    head_dim = embed_dim // num_heads       # num_heads * head_dim = embed_dim, 每头注意力的维度
    scaling = float(head_dim) ** -0.5       # 公式中的d^(-1/2)
    q = q * scaling                         # (query_len, batch_size, k_dim * num_heads)

    # 只会在解码器中的Masked Multi-Head Attention中用到
    if attn_mask is not None:
        # (tgt_len, src_len) or (num_heads*batch_size, tgt_len, src_len)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # (1, tgt_len,src_len) 扩充维度
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        # 现在 atten_mask 的维度就变成了3D

    """ 第三阶段 计算注意力权重矩阵Z """
    # 因为前面是num_heads个头一起参与的计算，所以这里要交换QKV的0,1两个维度,以便多个样本同时计算。
    # contiguous: 将变量放到连续的内存进行运算
    q = q.contiguous(). \
        view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)    # (batch_size * num_heads,tgt_len,k_dim)
    k = k.contiguous(). \
        view(-1, bsz * num_heads, head_dim).transpose(0, 1)         # (batch_size * num_heads,src_len,k_dim)
    v = v.contiguous(). \
        view(-1, bsz * num_heads, head_dim).transpose(0, 1)         # (batch_size * num_heads,src_len,v_dim)
    # 这就num_heads个QK^T相乘后的注意力矩阵
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))           # (batch_size * num_heads, tgt_len, src_len)

    """ 第四阶段 进行相关掩码操作"""
    # 注意力掩码计算
    if attn_mask is not None:
        attn_output_weights += attn_mask    # (batch_size * num_heads, tgt_len, src_len), 注意力矩阵 + attention掩码
    # padding掩码计算, 同时进行维度扩充
    if key_padding_mask is not None:
        # (batch_size * num_heads, tgt_len, src_len) -> (batch_size, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # (batch_size, src_len) -> (batch_size, 1, 1, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        # (batch_size * num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    """ num_heads个Attention(Q,K,V)结果 """
    # 对注意力权重矩阵归一化
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)    # (batch_size * num_heads, tgt_len, src_len)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    # 计算多头注意力机制输出
    # Z = (batch_size * num_heads, tgt_len, src_len)  x  (batch_size * num_heads, src_len, v_dim)
    attn_output = torch.bmm(attn_output_weights, v)                 # -> # (batch_size * num_heads, tgt_len, v_dim)

    # 先transpose成 (tgt_len, batch_size* num_heads ,k_dim)
    # 再view成 (tgt_len, batch_size, num_heads * k_dim)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    # 这里就是多个z  线性组合成Z  (tgt_len, batch_size, embed_dim)
    Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return Z, attn_output_weights.sum(dim=1) / num_heads  # 将num_heads个注意力权重矩阵按对应维度取平均


class MyMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0., bias=True):
        """
        :param embed_dim: 词嵌入的维度，也就是前面的d_model参数，论文中的默认值为512
        :param n_heads:   多头注意力机制中多头的数量，论文默认值为 8
        :param bias:      最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        super(MyMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # d_model
        self.head_dim = embed_dim // n_heads
        self.k_dim = self.head_dim
        self.v_dim = self.head_dim
        self.n_heads = n_heads  # 注意力头数
        self.dropout = dropout
        assert self.head_dim * self.n_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件
        self.w_q = Parameter(torch.Tensor(embed_dim, embed_dim))  # embed_dim = k_dim * n_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.w_k = Parameter(torch.Tensor(embed_dim, embed_dim))  # embed_dim = k_dim * n_heads
        self.w_v = Parameter(torch.Tensor(embed_dim, embed_dim))  # embed_dim = v_dim * n_heads

        # Z = (QK^T * (d_model)^(-1/2)) * V
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads
        self.w_out = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中，编码时query, key, value 都是同一个输入，
        解码时 输入的部分也都是同一个输入，
        解码和编码交互时 key,value指的是 memory(encoder输出), query指的是tgt
        :param query: (tgt_len, batch_size, embed_dim), tgt_len表示目标序列长度
        :param key:   (src_len, batch_size, embed_dim), src_len表示源序列长度
        :param value: (src_len, batch_size, embed_dim), src_len表示源序列长度
        :param attn_mask: (tgt_len, src_len) or (n_heads * batch_size, tgt_len, src_len)
                        一般只在解码时使用，为了并行一次喂入所有解码部分的输入，所以要用mask来进行掩盖当前时刻之后的位置信息
        :param key_padding_mask: (batch_size, src_len), src_len表示源序列长度
        :return:
        attn_output: (tgt_len, batch_size, embed_dim)
        attn_output_weights: (batch_size, tgt_len, src_len)
        """
        return multi_head_attention_forward(query, key, value, self.n_heads,
                                            self.dropout, self.w_out.weight, self.w_out.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.w_q,
                                            k_proj_weight=self.w_k,
                                            v_proj_weight=self.w_v,
                                            attn_mask=attn_mask)


if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    num_head = 1
    src = torch.rand((src_len, batch_size, d_model))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [src_len, src_len]

    my_mh = MyMultiHeadAttention(embed_dim=d_model, n_heads=num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
