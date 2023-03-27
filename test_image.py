# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.20 21:47
# @Author  : James.T
# @File    : test_image.py

import torch
import torchvision

import arch
import utils

from PIL import Image
import numpy as np
import argparse
import time
import os
import sys

model_path = 'checkpoint_srresnet.pth'

if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser(description='Resolve one image')
    parser.add_argument("img_path", type=str, default=None, help="image path")
    args = parser.parse_args()

    lr_img_path = args.img_path
    folder, lr_img_name = os.path.split(lr_img_path)
    name, file_type = os.path.splitext(lr_img_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # open img
    try:
        lr_img = Image.open(lr_img_path, mode='r')
        lr_img.convert('RGB')
    except FileNotFoundError:
        print(f'no image found at {lr_img_path}')
        sys.exit()

    # img transform
    lr_img = utils.input_transforms(lr_img).unsqueeze(0).to(device)

    # load model
    try:
        checkpoint = torch.load(model_path)
    except FileNotFoundError:
        print(f'no model found at {model_path}')
        sys.exit()
    model = arch.SRGenerator(
        large_kernel_size=9,
        small_kernel_size=3,
        n_channels=64,
        n_blocks=16,
        scaling_factor=4,
    ).to(device)
    if 'model' in checkpoint.keys():
        model.srresnet.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['generator'])
    model.eval()

    with torch.no_grad():

        # forward
        t = time.perf_counter()
        sr_img = model(lr_img)
        time_cost = time.perf_counter() - t
        print(f'costs {time_cost} us')

        # anti transform
        sr_img = utils.out_transform(sr_img)
        out_path = os.path.join(folder, f'{name}_{4}x{file_type}')
        torchvision.utils.save_image(sr_img, out_path)
        print(f'sr image output at \'{out_path}\'')
