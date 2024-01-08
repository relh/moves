#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import cupy as cp
import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.data
from cupyx.scipy.ndimage import label
from kornia.filters import spatial_gradient
from kornia.geometry.epipolar import sampson_epipolar_distance
from PIL import Image, ImageEnhance
from torch import pca_lowrank
from torchvision import transforms
from torchvision.utils import save_image

from skimage.color import *  # lab2rgb

inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
normalize = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  ),
])

values = (
        ("Definite Background", cv2.GC_BGD),
        ("Probable Background", cv2.GC_PR_BGD),
        ("Definite Foreground", cv2.GC_FGD),
        ("Probable Foreground", cv2.GC_PR_FGD),
)


def input_multiplexer(image_path, args):
    with Image.open(image_path) as keyframe_im:
        if args.sharpness != 1.0:
            inp_image_enh = keyframe_im.resize(args.img_size[::-1], resample=PIL.Image.BILINEAR)
            enhancer = ImageEnhance.Sharpness(inp_image_enh)
            inp_image_enh = enhancer.enhance(args.sharpness)
            keyframe = normalize(np.array(inp_image_enh)).unsqueeze(0).cuda()
            keyframe = F.interpolate(keyframe, size=args.img_size, mode='bilinear')
        else:
            keyframe = normalize(np.array(keyframe_im)).unsqueeze(0).cuda()
    return keyframe


def output_multiplexer(keyframe, net, mesh_grids, args):
    output = net(keyframe)['e']
    if args.norm: output = F.normalize(output, dim=1)
    output = torch.nan_to_num(output)
    return output


def get_pixel_groups(this_labels, batch_size=4096):
    all_pixels = []
    for b in range(this_labels.shape[0]):
        label_unique = this_labels[b].unique()
        this_max = batch_size // len(label_unique)
        spatial_label_unique = dict([(int(cc), (this_labels[b, 0] == cc).nonzero()) for cc in label_unique])

        # need to sample batch_size from these, sort so can always get enough from last
        spatial_label_unique = sorted(spatial_label_unique.items(), key=lambda i: i[1].shape[0])
        pixels = {}
        pixels_remaining = batch_size
        for zzz, (cc, embed_group) in enumerate(spatial_label_unique):
            # how many points to choose from this group
            indices_to_sample = min(pixels_remaining, this_max, embed_group.shape[0])
            pixels_remaining -= indices_to_sample

            # points to choose from this group
            indices = random.sample(range(0, embed_group.shape[0]), indices_to_sample)
            pixels[cc] = embed_group[indices, :]

        if pixels_remaining > 0:
            indices = random.choices(range(0, embed_group.shape[0]), k=pixels_remaining)
            pixels[cc] = torch.cat((pixels[cc], embed_group[indices, :]), dim=0)
        all_pixels.append(pixels)
    return all_pixels


def build_group_inputs(this_pixels, that_pixels, this_embedding, that_embedding, args):
    # David's pairwise loss
    mlp_inputs = []
    mlp_labels = []

    for b in range(len(this_pixels)):
        # build batch
        mlp_input = []
        mlp_label = []

        for chosen_cc, chosen_pixels in this_pixels[b].items():
            # negative points from background pixels
            pixels, labels = [], []

            # find matching connected component to understand how many positive we can take (max 25%)
            that_amount = 0
            if chosen_cc in that_pixels[b].keys(): 
                pos_that_pixels = that_pixels[b][chosen_cc]
                that_amount = min(chosen_pixels.shape[0] // 4, pos_that_pixels.shape[0])
                indices = random.sample(range(0, pos_that_pixels.shape[0]), that_amount)
                pixels.append(that_embedding[b, :, pos_that_pixels[indices, 0], pos_that_pixels[indices, 1]])
                labels.append((torch.zeros(that_amount) if args.negbg else -torch.ones(that_amount)) if chosen_cc == 0 else torch.ones(that_amount))

            # how many total negative pixels do we need to supplement this
            neg_that_amount, neg_this_amount = 0, 0
            if 0 in that_pixels[b].keys():
                neg_that_pixels = that_pixels[b][0]
                neg_that_amount = min(chosen_pixels.shape[0] // 4, neg_that_pixels.shape[0])
                indices = random.sample(range(0, neg_that_pixels.shape[0]), neg_that_amount)
                pixels.append(that_embedding[b, :, neg_that_pixels[indices, 0], neg_that_pixels[indices, 1]])
                labels.append((torch.zeros(neg_that_amount) if args.negbg else -torch.ones(neg_that_amount)) if chosen_cc == 0 else torch.zeros(neg_that_amount))

            if 0 in this_pixels[b].keys(): 
                neg_this_pixels = this_pixels[b][0]
                neg_this_amount = min(chosen_pixels.shape[0] // 4, neg_this_pixels.shape[0])
                indices = random.sample(range(0, neg_this_pixels.shape[0]), neg_this_amount)
                pixels.append(this_embedding[b, :, neg_this_pixels[indices, 0], neg_this_pixels[indices, 1]])
                labels.append((torch.zeros(neg_this_amount) if args.negbg else -torch.ones(neg_this_amount)) if chosen_cc == 0 else torch.zeros(neg_this_amount))

            this_amount = chosen_pixels.shape[0] - that_amount - neg_that_amount - neg_this_amount
            indices = random.sample(range(0, chosen_pixels.shape[0]), this_amount)
            pixels.append(this_embedding[b, :, chosen_pixels[indices, 0], chosen_pixels[indices, 1]])
            labels.append((torch.zeros(this_amount) if args.negbg else -torch.ones(this_amount)) if chosen_cc == 0 else torch.ones(this_amount))

            mlp_input.append(torch.cat((this_embedding[b, :, chosen_pixels[:, 0], chosen_pixels[:, 1]], torch.cat(pixels, dim=1)), dim=0))
            mlp_label.append(torch.cat(labels))
        mlp_input = torch.cat(mlp_input, dim=1)
        mlp_label = torch.cat(mlp_label, dim=0)

        mlp_inputs.append(mlp_input)
        mlp_labels.append(mlp_label)

    # run through an MLP with inputs being two embeddings
    return torch.cat(mlp_inputs, dim=1).permute(1,0), torch.cat(mlp_labels, dim=0).long().cuda()


def build_assoc_inputs(this_people, this_pixels, this_embedding):
    mlp_inputs = []
    mlp_labels = []

    # before split ccs, check if any single cc has people components and non
    # if so, make positive batches
    # if people

    for b in range(len(this_pixels)):
        # build batch
        mlp_input = []
        mlp_label = []
        how_many = len(this_pixels[b].items())

        for chosen_cc, chosen_pixels in this_pixels[b].items():
            # negative points from background pixels
            # split chosen_cc into people and non people
            # batches are positive between people and non people
            # negatives come from background
            
            # need all pixels with hands to pair with non hands
            # need all pixels with nonhands to pair with hands
            chosen_people = (this_people[b, chosen_pixels[:,0], chosen_pixels[:,1]]).bool()
            people_choices = [int(x) for x in list(torch.where(chosen_people == True)[0])]
            nonpeople_choices = [int(x) for x in list(torch.where(chosen_people == False)[0])]
            pos_this_amount = min(len(people_choices), len(nonpeople_choices))
            neg_needed = chosen_pixels.shape[0] - (pos_this_amount*2)
            random.shuffle(people_choices)
            random.shuffle(nonpeople_choices)

            #if len(people_choices) > len(nonpeople_choices):
            #    ordered_choices = people_choices + nonpeople_choices
            #    paired_choices = nonpeople_choices[:pos_this_amount]
            #else:
            #    ordered_choices = nonpeople_choices + people_choices
            #    paired_choices = people_choices[:pos_this_amount]
            if pos_this_amount > 0:
                ordered_choices = people_choices[:pos_this_amount] + nonpeople_choices[-pos_this_amount:] + ((people_choices + nonpeople_choices)[:neg_needed])
                paired_choices = nonpeople_choices[:pos_this_amount] + people_choices[-pos_this_amount:]
            else:
                ordered_choices = ((people_choices + nonpeople_choices)[:neg_needed])
                paired_choices = []

            # how many total negative pixels do we need to supplement this
            other_neg_embeds = []
            for i, (other_cc, other_pixels) in enumerate(this_pixels[b].items()):
                if chosen_cc == other_cc: 
                    if len(people_choices) > 0:
                        indices = random.sample(people_choices, k=min(len(people_choices), neg_needed // how_many))
                    else:                   
                        indices = random.sample(nonpeople_choices, k=min(len(nonpeople_choices), neg_needed // how_many))
                else:
                    indices = random.sample(range(0, other_pixels.shape[0]), k=min(other_pixels.shape[0], neg_needed // how_many))
                neg_needed -= len(indices)
                other_neg_embeds.append(this_embedding[b, :, other_pixels[indices, 0], other_pixels[indices, 1]])

            if neg_needed > 0:
                #for i, (other_cc, other_pixels) in enumerate(sorted(this_pixels[b].items(), key=lambda x: x[0])):
                for i, (other_cc, other_pixels) in enumerate(this_pixels[b].items()):
                    if chosen_cc == other_cc:
                        if len(people_choices) > 0: 
                            indices = random.choices(people_choices, k=neg_needed)
                        else: 
                            indices = random.choices(nonpeople_choices, k=neg_needed)
                    else:
                        indices = random.choices(range(0, other_pixels.shape[0]), k=neg_needed)
                    indices = random.choices(range(0, other_pixels.shape[0]), k=neg_needed)
                    neg_needed -= len(indices)
                    other_neg_embeds.append(this_embedding[b, :, other_pixels[indices, 0], other_pixels[indices, 1]])
                    if neg_needed <= 0: break
            other_neg_embeds = torch.cat(other_neg_embeds, dim=1)

            ordered_pixels = chosen_pixels[ordered_choices]
            ordered_embed = this_embedding[b, :, ordered_pixels[:, 0], ordered_pixels[:, 1]]

            pos_pixels = chosen_pixels[paired_choices] 
            pos_embed = this_embedding[b, :, pos_pixels[:, 0], pos_pixels[:, 1]]

            paired_embed = torch.cat((pos_embed, other_neg_embeds), dim=1)

            mlp_input.append(torch.cat((ordered_embed, paired_embed), dim=0))
            mlp_label.append(torch.cat((torch.ones(pos_this_amount * 2), torch.zeros(chosen_pixels.shape[0] - (pos_this_amount * 2)))))

        mlp_input = torch.cat(mlp_input, dim=1)
        mlp_label = torch.cat(mlp_label, dim=0)

        mlp_inputs.append(mlp_input)
        mlp_labels.append(mlp_label)

    # run through an MLP with inputs being two embeddings
    return torch.cat(mlp_inputs, dim=1).permute(1,0), torch.cat(mlp_labels, dim=0).long().cuda()


def d_flow(flow):
    # this method calculates the derivative of the flow
    _flow = einops.rearrange(flow, 'b h w c -> b c h w')
    _sg_flow = spatial_gradient(_flow, normalized=False)
    _d_flow = einops.rearrange([_sg_flow[:, 0, 0], _sg_flow[:, 1, 1]], 's b h w -> b s h w')
    return _d_flow


def build_corr_grid(flow, mesh_grids, args):
    # this grid will describe how this flow field warps to another
    #corr_grid = einops.repeat(mesh_grid, 'h w c -> b h w c', b=flow.shape[0])

    #newX, newY = X+dx, Y+dy
    #reverseDx, reverseDy = ...
    corr_grid = mesh_grids.clone()
    corr_grid[:, :, :, 0] += (flow[:, :, :, 0] / 512)
    corr_grid[:, :, :, 1] += (flow[:, :, :, 1] / 288)
    return corr_grid


def build_rewarp_grid(mesh_grids, that_corr_grid, this_corr_grid):
    #einops.repeat(mesh_grid.clone(), 'h w c -> b c h w', b=that_corr_grid.shape[0])
    mesh_grid_s = einops.rearrange(mesh_grids.clone(), 'b h w c -> b c h w')

    #interp(reverseDx AT newX, newY)
    #| X - interp(reversed, X+dx, Y+dy) |
    re_warped_corr_grid = F.grid_sample(mesh_grid_s, that_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    re_warped_corr_grid = F.grid_sample(re_warped_corr_grid, this_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    re_warped_corr_grid = einops.rearrange(re_warped_corr_grid, 'b c h w -> b h w c')
    return re_warped_corr_grid


def cycle_inconsistent(flow, re_warped_corr_grid, that_corr_grid, args):
    # optical flow for visualizing -> (X,Y) EVERYWHERE
    vis_flow = einops.rearrange(flow.clone(), 'b h w c -> b c h w')

    # check sum diff and skip bad matches with weighting -- previously 10
    re_warped_flow = F.grid_sample(vis_flow.clone(), re_warped_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    cycle_inconsistent = (abs(vis_flow - re_warped_flow).sum(dim=1) > 10.0)
    return cycle_inconsistent


def connected_components(mask, args):
    # gets connected components for all masks above the pre-filtered threshold
    all_labels = []
    for b in range(mask.shape[0]):
        this_labels = [torch.zeros(mask[b].shape)]
        this_max = 0
        for v in mask[b].unique().tolist():
            if v == 0: continue

            # get unique mask pieces
            this_label = (torch.as_tensor(label(cp.asarray((mask[b] == v).int()))[0]))

            # limit to 250 pixels+
            this_label = cleanse_component(this_label)

            # limit to 10 ccs
            #this_label[this_label > 20] = 0

            this_label[this_label != 0] += this_max
            this_labels.append(this_label)
            this_max = this_label.max()

        all_labels.append(torch.stack(this_labels).max(dim=0).values)
    return torch.stack(all_labels).cuda()


def merge_component(this_labels, that_labels, that_corr_grid, this_cycle_inconsistent, that_cycle_inconsistent):
    # gets connected components for all masks above the pre-filtered threshold
    clean_this_labels = this_labels.clone().float()
    clean_this_labels[this_cycle_inconsistent != 0.0] = 0.0
    this_warped_labels = F.grid_sample(clean_this_labels.unsqueeze(1), that_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True).int().squeeze()
    this_warped_labels[that_cycle_inconsistent != 0.0] = 0.0

    # fill that labels
    for v in sorted(this_labels.unique().tolist(), reverse=True): 
        if v == 0: continue

        that_values = that_labels[(this_warped_labels == v) & (that_labels != 0)].unique().tolist()
        max_val = max([v] + that_values)

        this_labels[this_labels == v] = max_val
        this_warped_labels[this_warped_labels == v] = max_val
        for vv in that_values: that_labels[that_labels == vv] = max_val

    #that_labels = torch.stack((this_warped_labels, that_labels)).max(dim=0).values
    return that_labels


def cleanse_component(this_label, min_size=250):
    if len(this_label.shape) < 3:
        this_label = this_label.unsqueeze(0)

    for b in range(this_label.shape[0]):
        for v in this_label[b].unique().tolist():
            if (this_label[b] == v).sum() < min_size:
                this_label[b][this_label[b] == v] = 0.0
    return this_label.squeeze()


def rebase_components(this_label, that_label):
    real_this_label, real_that_label = this_label.clone(), that_label.clone()
    for i, val in enumerate(sorted(list(set(this_label.unique().tolist() + that_label.unique().tolist())))):
        real_this_label[this_label == val] = i
        real_that_label[that_label == val] = i
    return real_this_label, real_that_label


def fit_motion_model(mask, cycle_inconsistent, that_corr_grid, ransac, acceptable, mesh_grids, args):
    # fits a motion model between two frames, within mask pixels
    all_F_mats, all_inl = [], []
    for b in range(that_corr_grid.shape[0]):
        pts_mask = (~cycle_inconsistent[b] & mask[b])

        if pts_mask.sum() <= 8: pts_mask = mask[b].cuda()
        if pts_mask.sum() <= 8: pts_mask = torch.ones(mask[b].shape).bool().cuda()
        if pts_mask.sum() <= 8:
            all_F_mats.append(torch.zeros((F_mat.shape)).cuda())
            all_inl.append(torch.zeros((inl.shape)).cuda())

        ptsA = (mesh_grids[b][pts_mask]).float()
        ptsB = (that_corr_grid[b][pts_mask]).float()
        pts = random.sample(range(0, ptsA.shape[0]), min(2500, ptsA.shape[0]))

        F_mat, inlier_mask = ransac.forward(ptsA[pts], ptsB[pts])

        # ensure good fit
        errors = ransac.error_fn(mesh_grids[b], that_corr_grid[b], einops.repeat(F_mat, 'h w -> c h w', c=1))
        inl = (errors <= (ransac.inl_th * acceptable)).cuda()

        all_F_mats.append(F_mat)
        all_inl.append((inl & pts_mask).float().max(dim=0).values)

        pts_mask = ((~inl & mask[b]) & ~cycle_inconsistent[b]).cuda()
    return torch.stack(all_F_mats), torch.stack(all_inl).bool()


def epipolar_distance(that_corr_grid, this_F_mats, mesh_grids, args):
    # produces a h x w map of epipolar error for the given motion model
    sed = sampson_epipolar_distance(mesh_grids.flatten(1, 2), that_corr_grid.flatten(1, 2), this_F_mats).reshape((mesh_grids.shape[0], mesh_grids.shape[1], mesh_grids.shape[2]))

    for b in range(that_corr_grid.shape[0]):
        sed[b] -= sed[b].min()
        sed[b] /= sed[b].max()
        sed[b] -= sed[b].mean()
    return sed #einops.rearrange(torch.stack(seds), 'i b h w -> b i h w')


def segment_embeddings(embed, clust):
    real_segments = []
    shape = (embed.shape[2], embed.shape[3])
    embed = einops.rearrange(embed, 'b c h w -> b (h w) c').cuda()

    for b in range(embed.shape[0]):
        try:
            real_segments.append(torch.as_tensor(clust.fit_predict(embed[b]).reshape(shape)))
        except Exception as e:
            print(f'clustering failed! {str(e)}')
            new_segment = torch.zeros(shape)
            new_segment[shape[0]-1:shape[0]+1, shape[1]-1:shape[1]+1] = 1
            real_segments.append(new_segment)

    return torch.stack(real_segments).cuda()


def change_brightness(img):
    '''
    input: BGR from cv2
    output: BGR
    '''
    img = Image.fromarray(np.uint8((inv_normalize(img) * 255.0).permute(1, 2, 0).cpu().numpy()))#[:, : , ::-1])
    #image brightness enhancer
    enhancer = ImageEnhance.Brightness(img)
    factor = 0.4 #darkens the image
    im_output = enhancer.enhance(factor)
    im_output = np.asarray(im_output)[:, :, ::-1]
    return im_output


def pca_image(y, rank=3):
    y = y.clip(-1e9, 1e9)
    y = y.detach().permute(1, 2, 0)
    y_shape = y.shape
    # visualize
    try:
        pca_y = pca_lowrank(y.reshape(-1, y.shape[-1]).float(), rank, center=True)[0]
        pca_y = pca_y.reshape(y_shape[0], y_shape[1], rank)
    except:
        pca_y = torch.zeros(y_shape[0], y_shape[1], rank)

    y = torch.nan_to_num(pca_y).cpu().numpy()
    y = (((y - y.mean(axis=(0, 1))) / ((3 * y.std(axis=(0, 1))) + 1e-7)) + 0.5).clip(0, 1)
    #y = (y - y.min()) / (y.max() - y.min())
    #y = xyz2rgb(y) #np.roll(y, 2, 2))
    return torch.tensor(y)


def store_image(inp=None, label='features', option='save', iii=0, bb=0, args=None):
    # swiss army knife method for saving outputs condition on what type of output it is
    for b in range(bb):
        if option == 'save':
            if 'score' in label:
                save_image(inp[b].float().cpu(), f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
            elif 'association' in label:
                if inp[b].sum() != 0:
                    save_image((inp[b] / inp[b].max()).cpu(), f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
            else:
                save_image((inp[b] / inp[b].max()).cpu(), f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
        elif option == 'overlay':
            mask, rgb = inp
            rgb_dimmed = change_brightness(rgb[b])
            this_mask = (cv2.cvtColor(np.float32(mask[b].cpu().numpy()), cv2.COLOR_GRAY2RGB) * (255, 0, 0)).astype(np.uint8)
            final_image = cv2.addWeighted(rgb_dimmed, 1.0, this_mask, 0.7, 0)
            cv2.imwrite(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', final_image)
        elif option == 'pca':
            # combo pca
            combo_pca = einops.rearrange(pca_image(torch.nan_to_num(torch.cat([x[b] for x in inp], dim=-1))), 'h w c -> c h w').cpu()
            for i, l in zip(range(len(inp)), label):
                save_image(combo_pca[:, :, i * args.embed_size[1]:(i + 1) * args.embed_size[1]], f'{args.experiment_path}/outputs/{iii}_{b}_{l}.png', format='png')
        elif option == 'flow':
            import flow_vis
            flow_vis_out = flow_vis.flow_to_color(torch.nan_to_num(inp[b]).cpu().numpy(), convert_to_bgr=False)
            flow_vis_out = Image.fromarray(np.uint8(flow_vis_out))
            flow_vis_out.save(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png')
        elif option == 'rgb':
            im = Image.fromarray(np.uint8((inv_normalize(inp[b]) * 255.0).permute(1, 2, 0).cpu().numpy()))
            im.save(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
        else:
            if 'bwr' in option:
                plt.imsave(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', (inp[b]).float().cpu(), cmap=option, vmin=-100, vmax=100)
            else:
                plt.imsave(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', (inp[b] / inp[b].max()).float().cpu(), cmap=option)
