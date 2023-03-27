# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.17 19:32
# @Author  : James.T
# @File    : train_srgan.py

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import utils
import arch
import datasets
import task_manager

import sys
import os
import argparse
import time
from tqdm import tqdm

if __name__ == '__main__':

    # arg
    parser = argparse.ArgumentParser(description='train SRGAN')
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
    task_path = task_path_dict['task_path']
    log_path = task_path_dict['log_path']
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

    # generator hyper-parameters
    large_kernel_size_g = task_config['generator_config']['large_kernel_size_g']
    small_kernel_size_g = task_config['generator_config']['small_kernel_size_g']
    n_channels_g = task_config['generator_config']['n_channels_g']
    n_blocks_g = task_config['generator_config']['n_blocks_g']
    generator_weight_initial = task_config['generator_config']['generator_weight_initial']

    # discriminator hyper-parameters
    kernel_size_d = task_config['discriminator_config']['kernel_size_d']
    n_channels_d = task_config['discriminator_config']['n_channels_d']
    n_blocks_d = task_config['discriminator_config']['n_blocks_d']
    fc_size_d = task_config['discriminator_config']['fc_size_d']

    # perceptual loss hyper-parameters
    vgg19_i = task_config['perceptual_config']['vgg19_i']  # VGG19网络第i个池化层
    vgg19_j = task_config['perceptual_config']['vgg19_j']  # VGG19网络第j个卷积层
    beta = task_config['perceptual_config']['beta']  # 判别损失乘子

    # learning hyper-parameters
    total_epochs = task_config['hyper_params']['total_epochs']
    batch_size = task_config['hyper_params']['batch_size']
    lr_initial = task_config['hyper_params']['lr_initial']
    lr_decay_gamma = task_config['hyper_params']['lr_decay_gamma']
    lr_milestones = task_config['hyper_params']['lr_milestones']

    # device settings
    n_gpu = task_config['others']['n_gpu']
    worker = task_config['others']['worker']

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nTrain: using {device} device\n')

    cudnn.benchmark = True  # 对卷积进行加速

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    start_time = time.time()

    # ------------------------ model ------------------------ #

    # Generator
    generator = arch.SRGenerator(
        large_kernel_size=large_kernel_size_g,
        small_kernel_size=small_kernel_size_g,
        n_channels=n_channels_g,
        n_blocks=n_blocks_g,
        scaling_factor=scaling_factor,
    )

    # Discriminator
    discriminator = arch.SRDiscriminator(
        kernel_size=kernel_size_d,
        n_channels=n_channels_d,
        n_blocks=n_blocks_d,
        fc_size=fc_size_d,
    )

    # content network
    content_network = arch.TruncatedVGG19(
        i=vgg19_i,
        j=vgg19_j,
    )
    content_network.eval()

    # ------------------------ loss ------------------------ #

    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # ------------------ load checkpoints ------------------ #

    # if resume, restart from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Checkpoint found, loading...')
            checkpoint = torch.load(args.resume)

            # load model weights
            start_epoch = checkpoint['epoch'] + 1
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])

        else:
            raise FileNotFoundError(f'No checkpoint found at \'{args.resume}\'')

    # if no resume, train from pretrained SRResNet
    else:
        if generator_weight_initial:
            try:
                initial_w_g = torch.load(generator_weight_initial)
                print('loading generator initial weights')
                generator.srresnet.load_state_dict(initial_w_g['model'])
            except FileNotFoundError:
                print(f'no file found at {generator_weight_initial}')

        else:
            print('WARNING: train generator from scratch')
        start_epoch = 0

    # ------------------ to device ------------------ #

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    content_network = content_network.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # ------------------------ optimizer and scheduler ------------------------ #

    # for generator
    optimizer_g = torch.optim.Adam(params=generator.parameters(), lr=lr_initial)
    scheduler_g = torch.torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer_g,
        milestones=lr_milestones,
        gamma=lr_decay_gamma,
        verbose=False,
    )

    # for discriminator
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=lr_initial)
    scheduler_d = torch.torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer_d,
        milestones=lr_milestones,
        gamma=lr_decay_gamma,
        verbose=False,
    )

    # if resume load optimizer
    if args.resume:
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        scheduler_d.load_state_dict(checkpoint['scheduler_d'])

    # ------------------------ multi gpu ------------------------ #
    is_multi_gpu = torch.cuda.is_available() and n_gpu > 1
    if is_multi_gpu:
        generator = nn.DataParallel(generator, device_ids=list(range(n_gpu)))  # 之后的项目应该用 nn.DistributedDataParallel
        discriminator = nn.DataParallel(discriminator, device_ids=list(range(n_gpu)))

    # ------------------------ dataset ------------------------ #

    train_dataset = datasets.SRDataset(
        dataset_name=dataset_name,
        lr_target_type='imagenet-norm',
        hr_target_type='imagenet-norm',
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
    #  adversarial training
    # --------------------------------------------------------------------

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training started at {strtime} '))

    # --------------------------
    loss_epochs_g_content_list = utils.LossSaver()  # content loss
    loss_iters_g_content_list = utils.LossSaver()

    loss_epochs_g_adversarial_list = utils.LossSaver()  # adversarial loss
    loss_iters_g_adversarial_list = utils.LossSaver()

    loss_epochs_d_list = utils.LossSaver()  # discriminator loss
    loss_iters_d_list = utils.LossSaver()

    writer = SummaryWriter()  # tensorboard

    total_bar = tqdm(range(start_epoch, total_epochs), desc='[Total Progress]')
    for epoch in total_bar:

        generator.train()
        discriminator.train()  # 另一方训练的时候 不需要 .eval()

        # --------------------------
        loss_epoch_g_content = utils.AverageMeter()
        loss_epoch_g_adversarial = utils.AverageMeter()
        loss_epoch_d = utils.AverageMeter()

        per_epoch_bar = tqdm(train_dataloader, leave=False)
        for lr_imgs, hr_imgs in per_epoch_bar:

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # --------------------- train generator ---------------------

            # generator sr
            sr_imgs = generator(lr_imgs)
            sr_imgs = utils.imagenet_norm_transform(
                utils.out_transform(sr_imgs)
            )  # convert to imagenet norm

            # content loss
            sr_imgs_content_feature = content_network(sr_imgs)
            hr_imgs_content_feature = content_network(hr_imgs).detach()

            content_loss_g = content_loss_criterion(
                sr_imgs_content_feature,
                hr_imgs_content_feature,
            )

            # adversarial loss
            sr_discriminated = discriminator(sr_imgs)
            adversarial_loss_g = adversarial_loss_criterion(
                sr_discriminated,
                torch.ones_like(sr_discriminated),
            )

            # perceptual loss
            perceptual_loss = content_loss_g + beta * adversarial_loss_g

            # backward
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # update generator weights
            optimizer_g.step()

            # record loss --------------------------
            loss_iters_g_content_list.append(content_loss_g.item())
            loss_epoch_g_content.update(content_loss_g.item(), lr_imgs.shape[0])
            loss_iters_g_adversarial_list.append(adversarial_loss_g.item())
            loss_epoch_g_adversarial.update(adversarial_loss_g.item(), lr_imgs.shape[0])

            # --------------------- train discriminator ---------------------

            # generator labels
            sr_discriminated = discriminator(sr_imgs.detach())
            hr_discriminated = discriminator(hr_imgs)

            # adversarial_loss
            adversarial_loss_d = \
                adversarial_loss_criterion(
                    sr_discriminated, torch.zeros_like(sr_discriminated)
                ) + \
                adversarial_loss_criterion(
                    hr_discriminated, torch.ones_like(hr_discriminated)
                )

            # backward
            optimizer_d.zero_grad()
            adversarial_loss_d.backward()

            # update discriminator weights
            optimizer_d.step()

            # record loss --------------------------
            loss_iters_d_list.append(adversarial_loss_d.item())
            loss_epoch_d.update(adversarial_loss_d.item(), lr_imgs.shape[0])

        # update scheduler
        scheduler_g.step()
        scheduler_d.step()

        # record img --------------------------
        lr_gird = make_grid(lr_imgs[:16, :3, :, :].cpu(), nrow=4, normalize=True)
        hr_gird = make_grid(hr_imgs[:16, :3, :, :].cpu(), nrow=4, normalize=True)
        sr_gird = make_grid(sr_imgs[:16, :3, :, :].cpu(), nrow=4, normalize=True)
        utils.save_img_to_file(lr_gird, os.path.join(record_path, f'epoch_{epoch}_lr.png'))
        utils.save_img_to_file(hr_gird, os.path.join(record_path, f'epoch_{epoch}_hr.png'))
        utils.save_img_to_file(sr_gird, os.path.join(record_path, f'epoch_{epoch}_sr.png'))
        writer.add_image(f'{task_id}/epoch_{epoch}_lr', lr_gird)
        writer.add_image(f'{task_id}/epoch_{epoch}_hr', hr_gird)
        writer.add_image(f'{task_id}/epoch_{epoch}_sr', sr_gird)

        del lr_imgs, hr_imgs, sr_imgs, lr_gird, hr_gird, sr_gird, \
            sr_imgs_content_feature, hr_imgs_content_feature, \
            hr_discriminated, sr_discriminated

        # record loss --------------------------
        loss_epochs_g_content_list.append(loss_epoch_g_content.val)
        loss_epochs_g_adversarial_list.append(loss_epoch_g_adversarial.val)
        loss_epochs_d_list.append(loss_epoch_d.val)
        writer.add_scalar(f'{task_id}/Loss_Content_G', loss_epoch_g_content.val, epoch)
        writer.add_scalar(f'{task_id}/Loss_Adversarial_G', loss_epoch_g_adversarial.val, epoch)
        writer.add_scalar(f'{task_id}/Loss_D', loss_epoch_d.val, epoch)

        # save model
        torch.save(
            {
                'epoch': epoch,
                'generator':
                    generator.module.state_dict() if is_multi_gpu else generator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'scheduler_g': scheduler_g.state_dict(),
                'discriminator':
                    discriminator.module.state_dict() if is_multi_gpu else discriminator.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'scheduler_d': scheduler_d.state_dict(),
            },
            os.path.join(checkpoint_path, f'checkpoint_srgan.pth'),
        )

    # save loss file
    loss_iters_g_content_list.save_to_file(os.path.join(record_path, f'loss_iters_g_content_{start_epoch}_{total_epochs-1}.npy'))
    loss_epochs_g_content_list.save_to_file(os.path.join(record_path, f'loss_epochs_g_content_{start_epoch}_{total_epochs - 1}.npy'))

    loss_iters_g_adversarial_list.save_to_file(os.path.join(record_path, f'loss_iters_g_adversarial_{start_epoch}_{total_epochs - 1}.npy'))
    loss_epochs_g_adversarial_list.save_to_file(os.path.join(record_path, f'loss_epochs_g_adversarial_{start_epoch}_{total_epochs - 1}.npy'))

    loss_iters_d_list.save_to_file(os.path.join(record_path, f'loss_iters_d_{start_epoch}_{total_epochs - 1}.npy'))
    loss_epochs_d_list.save_to_file(os.path.join(record_path, f'loss_epochs_d_{start_epoch}_{total_epochs - 1}.npy'))

    writer.close()

    # print time
    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total training costs {cost_time}')
