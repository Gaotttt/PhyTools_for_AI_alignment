# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.models.layers import DropPath
from network.AFNO_Block import AFNO1D, Mlp

class AFNO(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO1D(hidden_size=dim,
                             num_blocks=num_blocks,
                             sparsity_threshold=sparsity_threshold,
                             hard_thresholding_fraction=hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

initi = torch.zeros(20,50,64)
input = initi.transpose(1,0)
afno = AFNO(dim=64)
output = afno(input)
print(output.size())
