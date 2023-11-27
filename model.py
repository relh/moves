#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from hr_net import config
from hr_net.hrnet import get_seg_model


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.g = 1.0
            self.b = 0.0

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class MOVES_Model(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        embed_dim = args.embed_size[0]
        self.args = args

        config.defrost()
        config.merge_from_file('./hr_net/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
        config.merge_from_list(['DATASET.NUM_CLASSES', embed_dim, 'TRAIN.IMAGE_SIZE', args.img_size, 'TRAIN.BASE_SIZE', args.img_size[0]])
        config.freeze()
        backbone = get_seg_model(config)
        self.backbone = backbone

        if 'dd' in args.target:
            # build a 3-layer MLP for comparing embeddings
            self.compare_nn = nn.Sequential(
              nn.Linear(embed_dim * 2, embed_dim * 2, bias=False),
              nn.BatchNorm1d(embed_dim * 2),
              nn.ReLU(inplace=True), # first layer
              nn.Linear(embed_dim * 2, embed_dim, bias=False),
              nn.BatchNorm1d(embed_dim),
              nn.ReLU(inplace=True), # first layer
              nn.Linear(embed_dim, 2, bias=False))

        if 'aa' in args.target:
            # build a 3-layer MLP for comparing embeddings
            self.compare_assoc = nn.Sequential(
              nn.Linear(embed_dim * 2, embed_dim * 2, bias=False),
              nn.BatchNorm1d(embed_dim * 2),
              nn.ReLU(inplace=True), # first layer
              nn.Linear(embed_dim * 2, embed_dim, bias=False),
              nn.BatchNorm1d(embed_dim),
              nn.ReLU(inplace=True), # first layer
              nn.Linear(embed_dim, 2, bias=False))

    def forward(self, x):
        z = self.backbone(x)
        z = F.interpolate(z, size=self.args.embed_size, mode='bilinear')
        return {'e': z}

if __name__ == "__main__":
    dummy = MOVES_Model()
