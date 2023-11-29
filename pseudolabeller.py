#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import re
from multiprocessing import Process, Queue, cpu_count, set_start_method

import albumentations as albu
import cv2
import numpy as np
import torch
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad
from mmflow.apis import inference_model, init_model
from people_segmentation.pre_trained_models import create_model
from PIL import Image

set_start_method('spawn', force=True)


def crop_and_resize(frame, target_size):
    h, w = frame.shape[:2]
    # Determine the size of the square crop
    crop_size = min(h, w)

    # Calculate crop coordinates (center crop)
    startx = w // 2 - (crop_size // 2)
    starty = h // 2 - (crop_size // 2)    

    # Crop the largest square possible from the center
    crop = frame[starty:starty+crop_size, startx:startx+crop_size]

    # Resize the cropped square to the target size
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    return resized


def get_frame_pairs(frame_folder):
    frames = sorted(os.listdir(frame_folder))
    frame_dict = {}

    # Regular expression to extract video name and frame index
    pattern = re.compile(r"(.+)_frame(\d+)\.png")

    for frame in frames:
        if frame.endswith(".png"):
            match = pattern.match(frame)
            if match:
                video_name, frame_index = match.groups()
                if video_name not in frame_dict:
                    frame_dict[video_name] = []
                frame_dict[video_name].append((int(frame_index), frame))

    for video_frames in frame_dict.values():
        sorted_frames = sorted(video_frames)  # Sort by frame index
        for i in range(len(sorted_frames) - 1):
            yield (os.path.join(frame_folder, sorted_frames[i][1]),
                   os.path.join(frame_folder, sorted_frames[i + 1][1]))


def process_videos(video_folder='./videos/', frame_folder='./workspace/frames/', frame_size=(512, 512)):
    print(f'extracting frames from {video_folder} to {frame_folder}..')
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder, exist_ok=True)

    videos = sorted([x for x in os.listdir(video_folder) if x.endswith((".mp4", ".avi", ".mov"))])

    for iii, filename in enumerate(videos):
        if iii % 10 == 0:
            print(f'processing video {iii} / {len(videos)}')

        path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 0.5)  # 0.5 seconds between frames

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                cropped_resized_frame = crop_and_resize(frame, frame_size)
                cv2.imwrite(os.path.join(frame_folder, f"{filename}_frame{count}.png"), cropped_resized_frame)
            count += 1
        cap.release()


def process_people(frame_folder='./workspace/frames/', people_folder='./workspace/people/'):
    print(f'estimating people from frames in {frame_folder}, saving in {people_folder}..')
    if not os.path.exists(people_folder):
        os.makedirs(people_folder, exist_ok=True)

    print(f'initializing ternaus model..')
    model = create_model("Unet_2020-07-20").to(0)
    model.eval()

    all_frames = os.listdir(frame_folder)
    for iii, filename in enumerate(all_frames):
        frame_path = os.path.join(frame_folder, filename)

        if iii % 100 == 0:
            print(f'estimating people in frame {iii} / {len(all_frames)}.. {frame_path}')
        frame = cv2.imread(frame_path)
        image = np.uint8(frame)

        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0).to(0)
        prediction = model(x)[0][0]
        mask = (prediction > 0).bool()

        x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
        height, width = mask.shape[:2]
        mask = mask[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
        mask = mask.cpu().numpy()

        mask = Image.fromarray(np.asarray(mask))
        output_path = os.path.join(people_folder, os.path.basename(frame_path))
        mask.save(output_path, format='PNG')


def flow_worker_process(gpu_index, tasks):
    # you need to have setup mmflow and run the following command to run this script:
    #`mim download mmflow --config raft_8x2_100k_flyingthings3d_sintel_368x768`
    config_file = '~/.cache/mim/raft_8x2_100k_flyingthings3d_sintel_368x768.py'
    checkpoint_file = '~/.cache/mim/raft_8x2_100k_flyingthings3d_sintel_368x768.pth'

    print(f'initializing raft model.. {checkpoint_file}')
    model = init_model(config_file, checkpoint_file, device=f'cuda:{gpu_index}')

    for iii, (now_frame, next_frame) in enumerate(tasks):
        if iii % 100 == 0:
            print(f'estimating fwd/bck flow in pair {iii} / {len(tasks)}.. {now_frame} <-> {next_frame}')

        rgb_now_frame = cv2.imread(now_frame)
        rgb_future_frame = cv2.imread(next_frame)

        fwd_flow = (inference_model(model, rgb_now_frame, rgb_future_frame)).astype(np.float16)
        np.savez_compressed(now_frame.replace('frames', 'fwd_flow'), fwd_flow)

        bck_flow = (inference_model(model, rgb_future_frame, rgb_now_frame)).astype(np.float16)
        np.savez_compressed(now_frame.replace('frames', 'bck_flow'), bck_flow)


def process_flow(frame_folder='./workspace/frames/'):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Ensure that GPUs are properly installed.")

    print(f'making flow output folders..')
    if not os.path.exists(frame_folder.replace('frames', 'fwd_flow')):
        os.makedirs(frame_folder.replace('frames', 'fwd_flow'), exist_ok=True)

    if not os.path.exists(frame_folder.replace('frames', 'bck_flow')):
        os.makedirs(frame_folder.replace('frames', 'bck_flow'), exist_ok=True)

    # build frame cache
    pfc = [(now_frame, next_frame) for now_frame, next_frame in get_frame_pairs(frame_folder)]

    print(f'saving paired frame cache to disk..') 
    with open('./paired_frame_cache.pkl', 'wb') as pfc_file:
        pickle.dump(pfc, pfc_file, pickle.HIGHEST_PROTOCOL)

    # Split tasks into chunks and start worker processes
    total_tasks = len(pfc)
    chunk_size = (total_tasks + num_gpus - 1) // num_gpus  # Calculate chunk size
    processes = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, total_tasks)
        chunk = pfc[start:end]
        p = Process(target=flow_worker_process, args=(i, chunk))
        processes.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='./videos/', help="Path to the video folder")
    args = parser.parse_args()

    print(f'video folder is {args.video_path}')
    process_videos(args.video_path, './workspace/frames/')
    process_people('./workspace/frames/', './workspace/people/')
    process_flow('./workspace/frames/')
