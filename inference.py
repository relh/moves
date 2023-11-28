#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms

from functions import (input_multiplexer, output_multiplexer, save_image,
                       segment_embeddings)
from write import write_inference_html


def inference(images_path, net, clust, args):
    inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
      ),
    ])

    with torch.set_grad_enabled(False):
        # x,y grid for optical flow
        mesh_grids = torch.stack(list(torch.meshgrid(torch.linspace(-1, 1, steps=args.img_size[1]), torch.linspace(-1, 1, args.img_size[0]), indexing='xy')))
        mesh_grids = einops.repeat(mesh_grids, 'c h w -> b h w c', b=1).cuda(non_blocking=True)

    # load RGB
    for test_image in os.listdir(images_path):
        keyframe = input_multiplexer(images_path + test_image, args)
        save_image(keyframe, f'{args.experiment_path}/test_outputs/{test_image}_image.png', option='rgb')

        # run forward
        output = output_multiplexer(keyframe, net, mesh_grids, args)

        # save pca
        pca_feat = pca_image(output[0])
        pca_feat = F.interpolate(einops.repeat(pca_feat.float(), 'h w c -> b c h w', b=1), size=args.img_size).squeeze()
        save_image(pca_feat, f'{args.experiment_path}/test_outputs/{test_image}_features.png')

        if args.norm:
            output = F.normalize(output, dim=1)
        output = torch.nan_to_num(output) 

        # cluster embeddings into segments
        segments = segment_embeddings(output[:, :min(args.embed_size[0], output.shape[1])], clust).float()
        segments = F.interpolate(einops.repeat(segments.float(), 'c h w -> b c h w', b=1), size=args.img_size).squeeze()
        plt.imsave(f'{args.experiment_path}/test_outputs/{test_image}_clusters.png', (segments / segments.max()).float().cpu(), cmap='nipy_spectral')

    write_inference_html(args)
    return segments
