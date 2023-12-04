#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import pickle
import random
import sys
from pathlib import Path

import cupy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler

from dataset import MOVES_Dataset
from model import MOVES_Model
from write import write_index_html


def setup(rank, args):
    args.rank = rank

    # --- initialize ddp ---
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    os.environ['NCCL_P2P_DISABLE'] = str(1)
    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.num_gpus, init_method='env://')
    torch.cuda.set_device(rank)
    cupy.cuda.Device(rank).use()

    # --- initialize network ---
    net = MOVES_Model(args).cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DistributedDataParallel(net, device_ids=[rank], output_device=rank)#, find_unused_parameters=True)

    # --- load filtered paired frames and split into train/val ---
    pfc = pickle.load(open('paired_frame_cache.pkl', 'rb'))
    train_frames, valid_frames = pfc[:int(len(pfc) * 0.8)], pfc[int(len(pfc) * 0.8):]

    # --- shuffle frames for visualization diversity ---
    random.Random(args.rank).shuffle(valid_frames)

    # --- make datasets, samplers, dataloaders ---
    train_data, valid_data = MOVES_Dataset(train_frames, people=args.people, train=True), MOVES_Dataset(valid_frames, people=args.people)
    train_sampler, valid_sampler = DistributedSampler(train_data, shuffle=True), DistributedSampler(valid_data, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size // args.num_gpus, num_workers=0, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size // args.num_gpus, num_workers=0, sampler=valid_sampler, drop_last=True)

    # --- load checkpoint ---
    def load(rank, net):
        net_mtime = datetime.datetime.fromtimestamp(60*60*24.0)
        if net == None: return None, net_mtime
        checkpoints = list(Path(args.experiment_path + 'models/').rglob("*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: float(str(x).split('/')[-1].split('_')[0]))
            if rank == 0: print(f'checkpoints! {checkpoints}')
            net_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(checkpoints[0]))
            load_dict = torch.load(checkpoints[0], map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
            net.load_state_dict(load_dict['model'])
            if rank == 0: print(f'loaded model! {str(checkpoints[0])}')
        return net, net_mtime

    # --- train model ---
    if args.train:
        from train import run_epoch
        print('training..')
        if args.load: net, net_mtime = load(rank, net)

        # --- setup optimizer ---
        optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.95), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
        rlrop = ReduceLROnPlateau(optimizer, 'min', patience=int(args.early_stopping / 2), factor=0.5)
        scaler = GradScaler()
        train_losses, valid_losses = [], []
        min_loss = sys.maxsize
        failed_epochs = 0

        for epoch in range(args.num_epoch):
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

            train_loss = run_epoch(train_loader, net, scaler, optimizer, epoch, args)
            valid_loss = run_epoch(valid_loader, net, None, None, epoch, args, is_train=False)

            dist.all_reduce(train_loss)
            dist.all_reduce(valid_loss)

            train_loss = float(train_loss) / args.num_gpus
            valid_loss = float(valid_loss) / args.num_gpus

            rlrop.step(valid_loss)
            print(valid_loss)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if valid_losses[-1] < min_loss:
                min_loss = valid_losses[-1]
                failed_epochs = 0
                model_out_path = args.experiment_path + 'models/' + '_'.join([str(float(min_loss)), str(epoch), 'model']) + '.ckpt'
                if args.rank == 0:
                    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, model_out_path)
                    print('saved model..\t\t\t {}'.format(model_out_path))
            else:
                failed_epochs += 1
                print('--> loss failed to decrease {} epochs..\t\t\tthreshold is {}, {} all..{}'.format(failed_epochs, args.early_stopping, valid_losses, min_loss))
            if failed_epochs > args.early_stopping: break

    if args.rank == 0: write_index_html(args)

    if args.load: net, net_mtime = load(rank, net)
    torch.set_grad_enabled(False)

    if args.visualize:
        valid_loss = run_epoch(valid_loader, net, None, None, epoch, args, is_train=False, visualize=True)

    if args.inference: 
        from inference import inference
        print('inference..')
        from cuml.cluster import HDBSCAN
        clust = HDBSCAN(min_samples=args.cluster_min_samples, min_cluster_size=args.cluster_min_size)
        inference('./test/', net, clust, args)

    dist.barrier()
    dist.destroy_process_group()
    print('ending')
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', dest='train', action='store_true', help='whether to train')
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='whether to visualize')
    parser.add_argument('--inference', dest='inference', action='store_true', help='whether to run inference')

    # experiment parameters
    parser.add_argument('--name', type=str, default='demo', help='exp name')
    parser.add_argument('--target', type=str, default='ddaa', help='people / objects / background / attention')
    parser.add_argument('--model', type=str, default='hrnet', help='simseg / convnext / hrnet')

    parser.add_argument('--img_size', type=int, default=(512, 512), help='what size embedding')
    parser.add_argument('--embed_size', type=int, default=(128, 128), help='what size embedding')

    parser.add_argument('--people', dest='people', action='store_true', help='using people or not')
    parser.add_argument('--merge', dest='merge', action='store_true', help='whether merge across time')
    parser.add_argument('--aug', dest='aug', action='store_true', help='using aug')
    parser.add_argument('--negbg', dest='negbg', action='store_true', help='using negatives in background')
    parser.add_argument('--norm', dest='norm', action='store_true', help='norm embeddings')

    # decoding parameters
    parser.add_argument('--cluster_min_samples', type=int, default=2, help='min core samples in a cluster')
    parser.add_argument('--cluster_min_size', type=int, default=90, help='min size of a cluster')
    parser.add_argument('--sharpness', type=float, default=3.0, help='cohesiv sharpness enhancement')

    # pseudolabel parameters
    parser.add_argument('--ransac_threshold', type=float, default=1e-5, help='ransac thresh')
    parser.add_argument('--pseudolabels', type=str, default='MOVES', help='whether to use MOVES / CIS / flow')
    parser.add_argument('--motion_model', type=str, default='kornia', help='whether to use kornia or cv2')

    # optimization parameters
    parser.add_argument('--lr', type=float, default=0.00015, help='what lr')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='what decay')
    parser.add_argument('--early_stopping', type=int, default=5, help='number of epochs before early stopping')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='whether to finetune')

    # machine parameters
    parser.add_argument('--load', dest='load', action='store_true', help='whether to load')
    parser.add_argument('--port', type=str, default='12489', help='what ddp port')
    parser.add_argument('--num_gpus', type=int, default=None, help='num_gpus')

    # training parameters
    parser.add_argument('--num_epoch', type=int, default=1000, help='num_epoch')
    parser.add_argument('--train_len', type=int, default=100000, help='number of samples per epoch')
    parser.add_argument('--valid_len', type=int, default=1000, help='number of outputs to write')

    parser.set_defaults(train=False)
    parser.set_defaults(visualize=False)
    parser.set_defaults(inference=True)
    parser.set_defaults(load=True)

    parser.set_defaults(people=False)
    parser.set_defaults(merge=True)
    parser.set_defaults(aug=True)
    parser.set_defaults(negbg=True)
    parser.set_defaults(norm=True)
    args = parser.parse_args()

    args.people = True if 'aa' in args.target else False
    args.name = '-'.join([args.name, args.model, str(args.target), str(args.lr), 'aug' if args.aug else 'noaug'])

    # --- machine setup: directories, one output path for everything ---
    if args.num_gpus is None:
        args.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    args.batch_size = args.num_gpus * 4
    print(f'args.num_gpus: {args.num_gpus}')

    args.experiment_path = f'./experiments/{args.name}/'
    print(args.experiment_path)
    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.experiment_path + 'models/', exist_ok=True)
    os.makedirs(args.experiment_path + 'outputs/', exist_ok=True)
    os.makedirs(args.experiment_path + 'test_outputs/', exist_ok=True)

    mp.spawn(setup, args=(args,), nprocs=args.num_gpus, daemon=True)
