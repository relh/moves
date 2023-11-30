#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

import cupy
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from cuml.cluster import HDBSCAN
from kornia.geometry.ransac import RANSAC
from torch import distributed as dist
from torch import nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from functions import (build_assoc_inputs, build_corr_grid, build_group_inputs,
                       build_rewarp_grid, cleanse_component,
                       connected_components, cycle_inconsistent,
                       epipolar_distance, fit_motion_model, get_pixel_groups,
                       merge_component, rebase_components, segment_embeddings,
                       store_image)
from write import write_index_html


def run_epoch(loader, net, scaler, optimizer, epoch, args, is_train=True, visualize=True):
    # --- synchronize workers ---
    dist.barrier()
    if is_train: loader.sampler.set_epoch(epoch)

    # --- initialize variables ---
    torch.set_grad_enabled(is_train)
    torch.backends.cudnn.benchmark = True
    cupy.cuda.Device(args.rank).use()

    pbar = tqdm(loader, ncols=200)
    epoch_len = args.train_len if is_train else args.valid_len 
    if epoch_len == -1: epoch_len = len(loader) * args.batch_size
    net.train() if is_train else net.eval()

    losses, loss, step = 0, 0, 0
    start = time.time()

    dd_ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    aa_ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=torch.tensor([0.1, 2.0]).cuda())

    ransac = RANSAC(model_type='fundamental', inl_th=(args.ransac_threshold), batch_size=4096, max_iter=5, confidence=0.99999, max_lo_iters=5)
    if not is_train and visualize:
        clust = HDBSCAN(min_samples=args.cluster_min_samples, min_cluster_size=args.cluster_min_size)

    with autocast(enabled=True):
        with torch.set_grad_enabled(False):
            # x,y grid for optical flow
            mesh_grids = torch.stack(list(torch.meshgrid(torch.linspace(-1, 1, steps=args.img_size[1]), torch.linspace(-1, 1, args.img_size[0]), indexing='xy')))
            mesh_grids = einops.repeat(mesh_grids, 'c h w -> b h w c', b=args.batch_size // args.num_gpus).cuda(non_blocking=True)

    # --- training loop ---
    for i, batch in enumerate(pbar):
        with autocast(enabled=True):
            if is_train: optimizer.zero_grad(set_to_none=True)
            ddl, aal = 0.0, 0.0

            with torch.set_grad_enabled(False):
                # ============= load input/output ===============
                now_future_flow = batch['now']['flow_n_f'].cuda(non_blocking=True)
                future_now_flow = batch['future']['flow_f_n'].cuda(non_blocking=True)

                now_rgb = batch['now']['rgb'].float().cuda(non_blocking=True)
                future_rgb = batch['future']['rgb'].float().cuda(non_blocking=True)

                now_people = batch['now']['people'].float().bool().cuda(non_blocking=True)
                future_people = batch['future']['people'].float().bool().cuda(non_blocking=True)

                if now_future_flow.shape[0] != mesh_grids.shape[0]: continue
                if future_now_flow.shape[0] != mesh_grids.shape[0]: continue

                # =============== correlation grids from flow ====================
                now_future_corr_grid = build_corr_grid(now_future_flow, mesh_grids, args)
                future_now_corr_grid = build_corr_grid(future_now_flow, mesh_grids, args)

                now_future_rewarp_corr_grid = build_rewarp_grid(mesh_grids, future_now_corr_grid, now_future_corr_grid)
                future_now_rewarp_corr_grid = build_rewarp_grid(mesh_grids, now_future_corr_grid, future_now_corr_grid)

                # =============== mask of cycle consistent warps ====================
                now_future_cycle_inconsistent = cycle_inconsistent(now_future_flow, now_future_rewarp_corr_grid, now_future_corr_grid, args)
                future_now_cycle_inconsistent = cycle_inconsistent(future_now_flow, future_now_rewarp_corr_grid, future_now_corr_grid, args)

        # =============== background and hand motion models ====================
        with torch.set_grad_enabled(False):
            now_future_F_mat, _ = fit_motion_model(~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
            future_now_F_mat, _ = fit_motion_model(~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)

            with autocast(enabled=True):
                # =============== mask of background and hand motion models ====================
                now_sed = epipolar_distance(now_future_corr_grid, now_future_F_mat, mesh_grids, args)
                future_sed = epipolar_distance(future_now_corr_grid, future_now_F_mat, mesh_grids, args)

                # =============== grouping based on epipolar distance ====================
                now_thresh = 0.0 
                now_outl = (now_sed > now_thresh).float()
                future_outl = (future_sed > 0.0).float()

                # =============== people involvement begins ====================
                # threshold epipolar distance and then group connected components
                now_labels, future_labels = connected_components(now_outl, args), connected_components(future_outl, args)
                now_cc_people, future_cc_people = connected_components(now_people.float(), args), connected_components(future_people.float(), args)

                # merge connected components suavely
                future_labels[future_labels != 0.0] += (now_labels.max() + 1)
                future_cc_people[future_cc_people != 0.0] += (now_cc_people.max() + 1)

                if args.merge:
                    now_labels = merge_component(future_labels, now_labels, now_future_corr_grid, future_now_cycle_inconsistent, now_future_cycle_inconsistent)
                    future_labels = merge_component(now_labels, future_labels, future_now_corr_grid, now_future_cycle_inconsistent, future_now_cycle_inconsistent)

                    now_cc_people = merge_component(future_cc_people, now_cc_people, now_future_corr_grid, future_now_cycle_inconsistent, now_future_cycle_inconsistent)
                    future_cc_people = merge_component(now_cc_people, future_cc_people, future_now_corr_grid, now_future_cycle_inconsistent, future_now_cycle_inconsistent)

                now_labels, future_labels = cleanse_component(now_labels), cleanse_component(future_labels)
                future_cc_people, now_cc_people = cleanse_component(future_cc_people), cleanse_component(now_cc_people)

                # split the connected components into people + objects
                now_labels, future_labels = rebase_components(now_labels, future_labels)
                max_label = max((now_labels.max(), future_labels.max()))
                now_split_labels, future_split_labels = now_labels.clone(), future_labels.clone()

                now_split_labels[now_labels != 0 & now_people] = (max_label + now_cc_people[now_labels != 0 & now_people])
                future_split_labels[future_labels != 0 & future_people] = (max_label + future_cc_people[future_labels != 0 & future_people])

                now_future_cycle_inconsistent = F.interpolate(einops.repeat(now_future_cycle_inconsistent.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()
                future_now_cycle_inconsistent = F.interpolate(einops.repeat(future_now_cycle_inconsistent.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()

                now_people = F.interpolate(einops.repeat(now_people.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()
                future_people = F.interpolate(einops.repeat(future_people.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()

                now_future_corr_grid = einops.rearrange(F.interpolate(einops.rearrange(now_future_corr_grid, 'b h w c -> b c h w'), size=args.embed_size), 'b c h w -> b h w c')
                future_now_corr_grid = einops.rearrange(F.interpolate(einops.rearrange(future_now_corr_grid, 'b h w c -> b c h w'), size=args.embed_size), 'b c h w -> b h w c')

                now_labels = F.interpolate(einops.repeat(now_labels.float(), 'b h w -> b c h w', c=1), size=args.embed_size)
                future_labels = F.interpolate(einops.repeat(future_labels.float(), 'b h w -> b c h w', c=1), size=args.embed_size)

                if 'dd' in args.target or 'aa' in args.target:
                    now_pixels = get_pixel_groups(now_labels)#, now_people, args)
                    future_pixels = get_pixel_groups(future_labels)#, future_people, args)

                now_weight = (1 - now_future_cycle_inconsistent.sum() / (2 * args.embed_size[0] * args.embed_size[1])) 
                future_weight = (1 - future_now_cycle_inconsistent.sum() / (2 * args.embed_size[0] * args.embed_size[1])) 

        # =============== forward ====================
        with autocast(enabled=True):
            n_o, f_o = net(now_rgb), net(future_rgb)

            if args.norm:
                n_o['e'], f_o['e'] = F.normalize(n_o['e'], dim=1), F.normalize(f_o['e'], dim=1)
            
            if 'dd' in args.target:
                moves_model = net.module 

                now_group_batch, now_group_labels = build_group_inputs(now_pixels, future_pixels, n_o['e'], f_o['e'], args)
                now_group_out = moves_model.compare_nn(now_group_batch)

                future_group_batch, future_group_labels = build_group_inputs(future_pixels, now_pixels, f_o['e'], n_o['e'], args)
                future_group_out = moves_model.compare_nn(future_group_batch)

            if 'aa' in args.target:
                moves_model = net.module 

                now_assoc_batch, now_assoc_labels = build_assoc_inputs(now_people, now_pixels, n_o['e']) 
                now_assoc_out = moves_model.compare_assoc(now_assoc_batch)

                future_assoc_batch, future_assoc_labels = build_assoc_inputs(future_people, future_pixels, f_o['e'])
                future_assoc_out = moves_model.compare_assoc(future_assoc_batch)

            if 'dd' in args.target:
                ddl += now_weight * torch.nan_to_num(dd_ce_loss(now_group_out, now_group_labels), nan=0.99)
                ddl += future_weight * torch.nan_to_num(dd_ce_loss(future_group_out, future_group_labels), nan=0.99)
                ddl = 10.0 * ddl

            if 'aa' in args.target:
                aal += now_weight * torch.nan_to_num(aa_ce_loss(now_assoc_out, now_assoc_labels), nan=0.99)
                aal += future_weight * torch.nan_to_num(aa_ce_loss(future_assoc_out, future_assoc_labels), nan=0.99)
                aal = 10.0 * aal

            if not is_train:
                now_segments = segment_embeddings(n_o['e'][:, :min(args.embed_size[0], n_o['e'].shape[1])], clust).float()
                future_segments = segment_embeddings(f_o['e'][:, :min(args.embed_size[0], f_o['e'].shape[1])], clust).float()

            loss = torch.tensor(0.0).cuda()
            if 'dd' in args.target: loss += ddl
            if 'aa' in args.target: loss += aal

            if loss != loss:
                print('NaNed!')
                sys.exit()

            local_step = ((i + 1) * args.batch_size)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ================ logging ===================
        losses += float(loss.detach())

        status = '{} epoch {}: itr {:<6}/ {}- {}- '.format('TRAIN' if is_train else 'VAL  ', epoch, local_step, epoch_len, args.name)
        if 'aa' in args.target: status += f'aal {aal:.4f}- '
        if 'dd' in args.target: status += f'ddl {ddl:.4f}- '
        status += 'avg l {:.4f}- lr {}- dt {:.4f}'.format(
          losses / (i + 1), # print batch loss and avg loss
          str(optimizer.param_groups[0]['lr'])[:7] if optimizer is not None else args.lr,
          time.time() - start) # batch time
        pbar.set_description(status)

        if not is_train:
            iii = i * args.num_gpus + args.rank
            b = now_rgb.shape[0]

            store_image([n_o['e'], f_o['e']], ['now_xy-rgb-with-feat', 'future_xy-rgb-with-feat'], 'pca', iii, b, args)
            store_image([n_o['e'][:, :args.embed_size[0]], f_o['e'][:, :args.embed_size[0]]], ['now_feat', 'future_feat'], 'pca', iii, b, args)

            if visualize:
                store_image(now_segments, 'now_clusters', 'nipy_spectral', iii, b, args)
                store_image(future_segments, 'future_clusters', 'nipy_spectral', iii, b, args)

            store_image(now_rgb, 'now_frame', 'rgb', iii, b, args)
            store_image(future_rgb, 'future_frame', 'rgb', iii, b, args)

            store_image(now_labels.squeeze(), 'now_pseudolabels', 'nipy_spectral', iii, b, args)
            store_image(future_labels.squeeze(), 'future_pseudolabels', 'nipy_spectral', iii, b, args)

            store_image(now_split_labels.squeeze(), 'now_pseudolabels-assoc', 'nipy_spectral', iii, b, args)
            store_image(future_split_labels.squeeze(), 'future_pseudolabels-assoc', 'nipy_spectral', iii, b, args)

            store_image(now_sed, 'now_sampson-error', 'save', iii, b, args)
            store_image(future_sed, 'future_sampson-error', 'save', iii, b, args)

            store_image(now_future_cycle_inconsistent.unsqueeze(1), 'now_pixels-inconsistent', 'save', iii, b, args)
            store_image(future_now_cycle_inconsistent.unsqueeze(1), 'future_pixels-inconsistent', 'save', iii, b, args)

            store_image([F.interpolate(now_people.unsqueeze(1), size=args.img_size).squeeze(), now_rgb], 'now_people', 'overlay', iii, b, args)
            store_image([F.interpolate(future_people.unsqueeze(1), size=args.img_size).squeeze(), future_rgb], 'future_people', 'overlay', iii, b, args)

            store_image(now_future_flow, 'now_xy-flow-all', 'flow', iii, b, args)
            store_image(future_now_flow, 'future_xy-flow-all', 'flow', iii, b, args)

        # =============== termination ================
        if local_step > epoch_len and epoch_len > 0: break
    avg_loss = losses / (i + 1 + 1e-3) # (i * loader.batch_size)

    if not is_train and args.rank == 0:
        write_index_html(args)

    return torch.as_tensor(avg_loss).cuda()


if __name__ == "__main__":
    pass
