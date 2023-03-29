# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.17 19:32
# @Author  : James.T
# @File    : train_srresnet_ddp.py

import sys

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import torch.distributed as dist

import utils
from arch import SRResNet
import datasets
import task_manager

import os
import argparse
import time
from tqdm import tqdm


def dist_worker(
        rank,
        world_size,
        task_path_dict,
        task_config,
        resume_path,
):
    torch.distributed.init_process_group(
        backend='nccl',  # 后端支持， nccl最好。
        init_method='tcp://localhost:23456',
        # 有多种方式，此处是用TCP协议，127.0.0.1为本机可写为localhost，23456为端口号，在1024-65535之间选一个没被占用的端口即可。
        world_size=world_size,
        # 进程数，1个进程可管理多张GPU；此处1个进程管1个GPU；
        rank=rank,
        # 当前进程编号；此处1个进程管1个GPU，所以也是本机GPU编号。
    )

    is_record_iter = False

    # path
    # task_path = task_path_dict['task_path']
    # log_path = task_path_dict['log_path']
    checkpoint_path = task_path_dict['checkpoint_path']
    record_path = task_path_dict['record_path']

    # config
    task_id = task_config['task_id']

    dataset_name = task_config['train_dataset_config']['train_dataset_name']
    crop_size = task_config['train_dataset_config']['crop_size']
    scaling_factor = task_config['train_dataset_config']['scaling_factor']

    large_kernel_size = task_config['generator_config']['large_kernel_size_g']
    small_kernel_size = task_config['generator_config']['small_kernel_size_g']
    n_channels = task_config['generator_config']['n_channels_g']
    n_blocks = task_config['generator_config']['n_blocks_g']

    total_epochs = task_config['hyper_params']['total_epochs']
    lr = task_config['hyper_params']['lr_initial']

    # per gpu batch size
    batch_size = task_config['hyper_params']['batch_size'] // task_config['others']['n_gpu']
    worker = task_config['others']['worker'] // task_config['others']['n_gpu']

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    # SRResNet
    model = SRResNet(
        large_kernel_size=large_kernel_size,
        small_kernel_size=small_kernel_size,
        n_channels=n_channels,
        n_blocks=n_blocks,
        scaling_factor=scaling_factor,
    )

    # model to device
    torch.cuda.set_device(rank)  # 1进程 1GPU，rank 可指 GPU 编号
    model.cuda(rank)

    # criterion
    criterion = nn.MSELoss().cuda(rank)

    start_epoch = 0

    # resume
    if resume_path:
        if os.path.isfile(resume_path):
            print(f'Checkpoint found, loading...')

            checkpoint = torch.load(
                resume_path,
                map_location={'cuda:%d' % 0: 'cuda:%d' % rank}
            )

            # load model weights
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
        else:
            raise FileNotFoundError(f'No checkpoint found at \'{resume_path}\'')

    else:
        print('train from scratch')

    # distribute
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    if resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # dataset
    train_dataset = datasets.SRDataset(
        dataset_name=dataset_name,
        lr_target_type='imagenet-norm',
        hr_target_type='[-1, 1]',
        crop_size=crop_size,
        scaling_factor=scaling_factor,
    )

    datasampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(datasampler is None),  # datasampler 自动 shuffle, 使用是 loader 不能 shuffle
        num_workers=worker,
        pin_memory=True,
        drop_last=True,  # 丢掉最后一个数量不够的 batch
    )

    # --------------------------------------------------------------------
    #  training
    # --------------------------------------------------------------------

    loss_epochs_list = utils.LossSaver()  # record loss per epoch
    if is_record_iter:
        loss_iters_list = utils.LossSaver()  # record loss per iter
    writer = SummaryWriter()  # tensorboard

    # 主进程控制记录与输出, 可以用一个函数包装一下
    if rank == 0:
        total_bar = tqdm(range(start_epoch, total_epochs), desc='[Total Progress]')
    else:
        total_bar = range(start_epoch, total_epochs)

    for epoch in total_bar:

        model.train()

        datasampler.set_epoch(epoch)  # enable shuffle

        loss_epoch = utils.AverageMeter()

        per_epoch_bar = tqdm(train_dataloader, leave=False) if rank == 0 else train_dataloader

        for lr_img, hr_img in per_epoch_bar:

            lr_img = lr_img.cuda(rank, non_blocking=True)
            hr_img = hr_img.cuda(rank, non_blocking=True)

            optimizer.zero_grad()

            # forward
            sr_img = model(lr_img)

            # loss
            loss = criterion(sr_img, hr_img)

            # backward
            loss.backward()

            # update weights
            optimizer.step()

            # record loss
            if is_record_iter:
                loss_iters_list.append(loss.item())
            loss_epoch.update(loss.item(), lr_img.shape[0])  # per epoch loss

        # cuda:0 record img change
        if rank == 0:
            lr_gird = make_grid(lr_img[:16, :3, :, :].cpu(), nrow=4, normalize=True)
            hr_gird = make_grid(hr_img[:16, :3, :, :].cpu(), nrow=4, normalize=True)
            sr_gird = make_grid(sr_img[:16, :3, :, :].cpu(), nrow=4, normalize=True)
            utils.save_img_to_file(lr_gird, os.path.join(record_path, f'epoch_{epoch}_lr.png'))
            utils.save_img_to_file(hr_gird, os.path.join(record_path, f'epoch_{epoch}_hr.png'))
            utils.save_img_to_file(sr_gird, os.path.join(record_path, f'epoch_{epoch}_sr.png'))
            writer.add_image(f'{task_id}/epoch_{epoch}_lr', lr_gird)
            writer.add_image(f'{task_id}/epoch_{epoch}_hr', hr_gird)
            writer.add_image(f'{task_id}/epoch_{epoch}_sr', sr_gird)
            writer.add_scalar(f'{task_id}/MSE_Loss', loss_epoch.val, epoch)  # write loss
        del lr_img, hr_img, sr_img, lr_gird, hr_gird, sr_gird

        # record loss
        loss_epochs_list.append(loss_epoch.val)

        # save model
        if rank == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),  # 不同 GPU 上 optimizer 应该是相同的
                },
                os.path.join(checkpoint_path, f'checkpoint_srresnet_gpu{rank}.pth'),
            )

    # save loss file
    loss_epochs_list.save_to_file(
        os.path.join(record_path, f'epoch_loss_{start_epoch}_{total_epochs - 1}_GPU{rank}.npy')
    )
    if is_record_iter:
        loss_iters_list.save_to_file(
            os.path.join(record_path, f'iter_loss_{start_epoch}_{total_epochs - 1}_GPU{rank}.npy')
        )

    if rank == 0:
        writer.close()


if __name__ == '__main__':

    # arg
    parser = argparse.ArgumentParser(description='train SRResNet')
    parser.add_argument("--taskid", "-id", type=str, default='', help="the task you want to train")
    parser.add_argument("--resume", "-r", type=str, default=None, help="the path of previous weights")
    args = parser.parse_args()

    resume_path = args.resume

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # load config and path
    tm = task_manager.TaskManager()
    task_path_dict_m = tm.get_task_path(args.taskid)
    print('\n{:-^52}\n'.format(' TASK CONFIG '))
    tm.display_task_config(args.taskid)
    task_config_m = tm.get_task_config(args.taskid)

    world_size = task_config_m['n_gpu']  # 使用多少个 GPU 进行训练

    # check is configured
    task_id = task_config_m['task_id']
    if not task_config_m['is_config']:
        print(f'task {task_id} not configured, if configured, set \'is_config\' to be true')
        sys.exit()

    # check device
    if not torch.cuda.is_available():
        print('Train: current device doesnt have available gpu')
        sys.exit()
    print(f'Train: using cuda device')

    cudnn.benchmark = True  # 加速卷积

    # --------------------------------------------------------------------
    #  multiprocessing
    # --------------------------------------------------------------------

    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23456"

    # time cost
    start_time = time.time()
    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training started at {strtime} '))

    torch.multiprocessing.spawn(
        dist_worker,
        nprocs=world_size,
        args=(world_size, task_path_dict_m, task_config_m, resume_path),
        # join=True,
    )

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total training costs {cost_time}')

# log(未实现)

