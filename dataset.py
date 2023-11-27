#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from autoaugment import TrivialAugmentWide


class MOVES_Dataset(Dataset):
    def __init__(self, pfc, people=False, train=False):
        self.pfc = pfc
        self.people = people
        self.transform = transforms.Compose(\
            ([TrivialAugmentWide()] if train else []) +
            [transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.pfc)

    def __getitem__(self, idx):
        now_frame, future_frame = self.pfc[idx]

        return {
          'now': {
            'frame': now_frame,
            'rgb': self.transform(Image.open(now_frame)),
            'flow_n_f': np.load(now_frame.replace('frames', 'fwd_flow') + '.npz', allow_pickle=True, mmap_mode='r')['arr_0'],
            'people': np.array(Image.open(now_frame.replace('frames', 'people'))) if self.people else np.zeros((512, 512))
          },
          'future': {
            'frame': future_frame,
            'rgb': self.transform(Image.open(future_frame)),
            'flow_f_n': np.load(now_frame.replace('frames', 'bck_flow') + '.npz', allow_pickle=True, mmap_mode='r')['arr_0'],
            'people': np.array(Image.open(future_frame.replace('frames', 'people'))) if self.people else np.zeros((512, 512))
          },
        }


if __name__ == "__main__":
    pfc = pickle.load(open('paired_frame_cache.pkl', 'rb'))
    moves_ds = MOVES_Dataset(pfc=pfc, people=False, train=False)
