#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/6/17 15:34
# software: PyCharm-utils
import copy
import torch.nn as nn


def get_clones(module: nn.Module, n_modules: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_modules)])
