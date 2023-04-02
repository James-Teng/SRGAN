# -*- coding: utf-8 -*-
# @Time    : 2023.4.2 15:29
# @Author  : James.T
# @File    : net_interp.py

import sys
import torch
from collections import OrderedDict

import arch

alpha = float(sys.argv[1])

net_PSNR_path = './task_record/20230331_2135_Friday_df2k/checkpoint/checkpoint_srresnet.pth'
net_SRGAN_path = './task_record/20230401_2017_Saturday_df2kg/checkpoint/checkpoint_srgan.pth'
net_interp_path = f'./interpolation/interp_{int(alpha*10):02d}.pth'

net_PSNR = torch.load(net_PSNR_path, map_location='cpu')['model']
net_SRGAN = torch.load(net_SRGAN_path, map_location='cpu')['generator']
net_interp = OrderedDict()

g = arch.SRGenerator()
g.load_state_dict(net_SRGAN)
net_SRGAN = g.srresnet.state_dict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_SRGAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)


# @InProceedings{wang2018esrgan,
#     author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
#     title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
#     booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
#     month = {September},
#     year = {2018}
# }
