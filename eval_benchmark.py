# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.21 21:52
# @Author  : James.T
# @File    : eval_benchmark.py

import torch
import torchvision

import arch
import utils
import datasets
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# from PIL import Image
# import numpy as np
import argparse
import time
import os
import sys
from tqdm import tqdm

scaling_factor = 4

if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser(description='Evaluations on Benchmark Datasets')
    parser.add_argument("model_path", type=str, default=None, help="model path")
    args = parser.parse_args()

    model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using {device} device')

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
    )

    if 'generator' in checkpoint.keys():
        model.load_state_dict(checkpoint['generator'])
    elif 'model' in checkpoint.keys():
        model.srresnet.load_state_dict(checkpoint['model'])
    else:
        model.srresnet.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # benchmark dataset
    eval_datasets = ["Set5", "Set14", "BSD100"]

    # create record file
    comment = time.strftime(f'%Y-%m-%d_%H-%M-%S', time.localtime())
    record_folder = os.path.join('test_result', comment)
    os.mkdir(record_folder)
    record_file_path = os.path.join(record_folder, f'eval_{comment}.txt')
    with open(record_file_path, 'x') as f:
        f.write(f'model path: {model_path}\n')
        f.write(f'eval time: {comment}\n')

    # --------------------------------------------------------------------
    #  evaluation
    # --------------------------------------------------------------------
    for dataset_name in eval_datasets:
        print(f'\nevaluate on {dataset_name}\n')

        os.mkdir(os.path.join(record_folder, dataset_name))

        # load dataset
        test_dataset = datasets.SRDataset(
            dataset_name=dataset_name,
            lr_target_type='imagenet-norm',
            hr_target_type='[-1, 1]',
            scaling_factor=scaling_factor,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        # PSNR and SSIM
        PSNRs = utils.AverageMeter()
        SSIMs = utils.AverageMeter()

        # time cost
        t = time.time()

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(tqdm(test_dataloader, leave=False, desc=f'{dataset_name}')):

                # transfer data to device
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # forward
                sr_imgs = model(lr_imgs)

                # save sr img
                sr_imgs = utils.out_transform(sr_imgs)
                out_path = os.path.join(record_folder, dataset_name, f'{i}.png')
                torchvision.utils.save_image(sr_imgs, out_path, quality=100)

                # eval on metrics
                sr_imgs_y = utils.y_channel_transform(
                    utils.out_transform(sr_imgs)
                ).squeeze()  # from [-1, 1] to yuv([0, 255])
                hr_imgs_y = utils.y_channel_transform(
                    utils.out_transform(hr_imgs)
                ).squeeze()  # from [-1, 1] to yuv([0, 255])

                psnr = peak_signal_noise_ratio(
                    hr_imgs_y.cpu().numpy(),
                    sr_imgs_y.cpu().numpy(),
                    data_range=255.
                )

                ssim = structural_similarity(
                    hr_imgs_y.cpu().numpy(),
                    sr_imgs_y.cpu().numpy(),
                    data_range=255.
                )

                PSNRs.update(psnr, lr_imgs.shape[0])
                SSIMs.update(ssim, lr_imgs.shape[0])

        # output to file
        t_cost_per = (time.time() - t) / len(test_dataset)
        print(f'PSNR {PSNRs.avg:.3f}')
        print(f'SSIM {SSIMs.avg:.3f}')
        print(f'costs {t_cost_per:.3f}s per image\n')
        with open(record_file_path, 'a') as f:
            f.write(f'\nevaluate on {dataset_name}\n')
            f.write(f'PSNR {PSNRs.avg:.3f}\n')
            f.write(f'SSIM {SSIMs.avg:.3f}\n')
            f.write(f'costs {t_cost_per:.3f} per image\n')
