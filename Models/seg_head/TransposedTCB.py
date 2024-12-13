import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Models.utils import TOPK
from .kan import KANLayer
import math



class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

class Decoder(nn.Module):
    def __init__(self, in_chans=[2048, 1024, 512],  in_features=[64, 256]):

        super().__init__()
        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([2048, 1024, 512, 256][i], 128), RB(128, 64), nn.Upsample(size=int(64))
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64), RB(64, 64)))

        self.PH = nn.Sequential(
            RB(64, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=256)
        self.ff1 = TOPK(in_chans=in_chans[0], embed_dim=in_chans[1], in_features=in_features[0])

        self.ff2 = TOPK(in_chans=in_chans[1], embed_dim=in_chans[2], in_features=in_features[1])

    def forward(self, x, x1):
        x[0] = self.ff1(x[0], x1[0])
        x[1] = self.ff2(x[1], x1[1])
        pyramid_emph = []
        for i, level in enumerate(x):
            pyramid_emph.append(self.LE[i](x[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l
        l = self.up_tosize(l)
        l = self.PH(l)

        return l


