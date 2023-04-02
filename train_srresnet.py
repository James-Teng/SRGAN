# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.17 19:32
# @Author  : James.T
# @File    : train_srresnet.py

import sys
import os
import argparse
import time
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import utils
from arch import SRResNet
import datasets
import task_manager

is_record_iter = False

if __name__ == '__main__':

    global is_record_iter

    # arg
    parser = argparse.ArgumentParser(description='train SRResNet')
    parser.add_argument("--taskid", "-id", type=str, default='', help="the task you want to train")
    parser.add_argument("--resume", "-r", type=str, default=None, help="the path of previous weights")
    args = parser.parse_args()

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # load config and path
    tm = task_manager.TaskManager()

    task_path_dict = tm.get_task_path(args.taskid)

    # path
    # task_path = task_path_dict['task_path']
    # log_path = task_path_dict['log_path']
    checkpoint_path = task_path_dict['checkpoint_path']
    record_path = task_path_dict['record_path']

    print('\n{:-^52}\n'.format(' TASK CONFIG '))
    tm.display_task_config(args.taskid)
    task_config = tm.get_task_config(args.taskid)

    # config
    task_id = task_config['task_id']
    if not task_config['is_config']:
        print(f'task {task_id} not configured, if configured, set \'is_config\' to be true')
        sys.exit()

    dataset_name = task_config['train_dataset_config']['train_dataset_name']
    crop_size = task_config['train_dataset_config']['crop_size']
    scaling_factor = task_config['train_dataset_config']['scaling_factor']

    large_kernel_size = task_config['generator_config']['large_kernel_size_g']
    small_kernel_size = task_config['generator_config']['small_kernel_size_g']
    n_channels = task_config['generator_config']['n_channels_g']
    n_blocks = task_config['generator_config']['n_blocks_g']

    total_epochs = task_config['hyper_params']['total_epochs']
    batch_size = task_config['hyper_params']['batch_size']
    lr = task_config['hyper_params']['lr_initial']

    n_gpu = task_config['others']['n_gpu']
    worker = task_config['others']['worker']

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nTrain: using {device} device\n')

    cudnn.benchmark = True  # 加速卷积

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    start_time = time.time()

    # SRResNet
    model = SRResNet(
        large_kernel_size=large_kernel_size,
        small_kernel_size=small_kernel_size,
        n_channels=n_channels,
        n_blocks=n_blocks,
        scaling_factor=scaling_factor,
    )

    # get resume file
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Checkpoint found, loading...')
            checkpoint = torch.load(args.resume)

            # load model weights
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])

        else:
            raise FileNotFoundError(f'No checkpoint found at \'{args.resume}\'')

    else:
        print('train from scratch')

    # to device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)  # filter(lambda p: p.requires_grad, model.parameters())

    # load optimizer
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # n gpu
    is_multi_gpu = torch.cuda.is_available() and n_gpu > 1
    if is_multi_gpu:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))  # 之后的项目应该用 nn.DistributedDataParallel

    # dataset
    train_dataset = datasets.SRDataset(
        dataset_name=dataset_name,
        lr_target_type='imagenet-norm',
        hr_target_type='[-1, 1]',
        crop_size=crop_size,
        scaling_factor=scaling_factor,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker,
        pin_memory=True,
    )

    # --------------------------------------------------------------------
    #  training
    # --------------------------------------------------------------------

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training started at {strtime} '))

    loss_epochs_list = utils.LossSaver()  # record loss per epoch
    if is_record_iter:
        loss_iters_list = utils.LossSaver()  # record loss per iter
    writer = SummaryWriter()  # tensorboard

    total_bar = tqdm(range(start_epoch, total_epochs), desc='[Total Progress]')
    for epoch in total_bar:

        model.train()

        loss_epoch = utils.AverageMeter()

        per_epoch_bar = tqdm(train_dataloader, leave=False)
        for lr_img, hr_img in per_epoch_bar:

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # forward
            sr_img = model(lr_img)

            # loss
            loss = criterion(sr_img, hr_img)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # record loss
            if is_record_iter:
                loss_iters_list.append(loss.item())
            loss_epoch.update(loss.item(), lr_img.shape[0])  # per epoch loss

        # record img change
        lr_gird = make_grid(lr_img[:16, :3, :, :].cpu(), nrow=4, normalize=True)
        hr_gird = make_grid(hr_img[:16, :3, :, :].cpu(), nrow=4, normalize=True)
        sr_gird = make_grid(sr_img[:16, :3, :, :].cpu(), nrow=4, normalize=True)
        utils.save_img_to_file(lr_gird, os.path.join(record_path, f'epoch_{epoch}_lr.png'))
        utils.save_img_to_file(hr_gird, os.path.join(record_path, f'epoch_{epoch}_hr.png'))
        utils.save_img_to_file(sr_gird, os.path.join(record_path, f'epoch_{epoch}_sr.png'))
        writer.add_image(f'{task_id}/epoch_{epoch}_lr', lr_gird)
        writer.add_image(f'{task_id}/epoch_{epoch}_hr', hr_gird)
        writer.add_image(f'{task_id}/epoch_{epoch}_sr', sr_gird)
        del lr_img, hr_img, sr_img, lr_gird, hr_gird, sr_gird

        # record loss
        loss_epochs_list.append(loss_epoch.val)
        writer.add_scalar(f'{task_id}/MSE_Loss', loss_epoch.val, epoch)

        # save model
        torch.save(
            {
                'epoch': epoch,
                'model': model.module.state_dict() if is_multi_gpu else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            os.path.join(checkpoint_path, f'checkpoint_srresnet.pth'),
        )

    # save loss file
    loss_epochs_list.save_to_file(os.path.join(record_path, f'epoch_loss_{start_epoch}_{total_epochs-1}.npy'))
    if is_record_iter:
        loss_iters_list.save_to_file(os.path.join(record_path, f'iter_loss_{start_epoch}_{total_epochs-1}.npy'))

    writer.close()

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total training costs {cost_time}')

# log(未实现)

