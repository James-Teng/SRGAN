import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms.functional as F
from torchvision import transforms
import torchvision

from collections.abc import Callable
from typing import Optional

import sys

plt.rcParams["savefig.bbox"] = 'tight'


# --------------
#   plotter
# --------------

# def show_tensor_image(imgs):
#     """
#     from pytorch tutorial
#     https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py
#     """
#     plt.figure()
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#         plt.show()


def show_pil_image(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    # plt.show()


def show_tensor_image(img):
    img = img.detach()
    img = F.to_pil_image(img)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    # plt.show()


def draw_loss_curve(loss):
    plt.figure()
    plt.plot(loss)
    plt.show()


# --------------------
#   log
# --------------------


# --------------------
#   训练中间过程记录
# --------------------

class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossSaver:
    """

    """
    def __init__(self):
        self.loss_list = []

    def reset(self):
        self.loss_list = []

    def append(self, loss):
        self.loss_list.append(loss)

    def to_np_array(self):
        return np.array(self.loss_list)

    def save_to_file(self, file_path):
        np_loss_list = np.array(self.loss_list)
        np.save(file_path, np_loss_list)


def load_loss_file(file_path):
    data = np.load(file_path)
    return data


def save_img_to_file(imgs, path):
    torchvision.utils.save_image(imgs, path)


# --------------------
#   image transforms
# --------------------

def compose_lr_transforms(
        img_type: str,
        scaling_factor: int
):
    """
    没有实现 img_type 转换
    downsample and to imagenet norm, applied when it is model's input
    """
    return transforms.Compose(
        [
            DownSampleResize(
                scaling_factor,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean and std
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compose_hr_transforms(
    img_type: str,
):
    """
    to [-1, 1], for supervision usage,
    对 target 进行变换
    """
    # 鲁棒性最好的做法应该是用 minmax norm
    if img_type == '[-1, 1]':
        return transforms.Compose(
            [
                transforms.ToTensor(),
                TargetNorm(),
            ]
        )
    elif img_type == 'imagenet-norm':
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean and std
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

# def compose_denormalize_transforms(
#         mean: list = (0.485, 0.456, 0.406),
#         std: list = (0.229, 0.224, 0.225)
# ):
#     """
#     un-normalize img, default is imagenet norm
#     """
#     mean_tensor = torch.tensor(mean)
#     std_tensor = torch.tensor(std)
#     return transforms.Normalize(list(-mean_tensor/std_tensor), list(1.0/std_tensor))


def input_transforms(img):
    """
    非数据集图像做超分的时候用
    """
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean and std
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return trans(img)


def imagenet_norm_transform(img):
    """
    from [0, 1] to imagenet norm
    """
    trans = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean and std
                std=[0.229, 0.224, 0.225],
            )
    return trans(img)


def out_transform(img):
    """
    from [-1, 1] to [0, 1]
    """
    img = (img + 1.) / 2.
    return img


def y_channel_transform(img):
    """
    from [0, 1] to yuv([0, 255])
    """
    # rgb_weights = torch.FloatTensor([0.299, 0.587, 0.114]).to(img.device)  # https://en.wikipedia.org/wiki/YUV
    # img = \
    #     torch.matmul(
    #         255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],  # remove border according to paper
    #         rgb_weights
    #     )
    # metrics 仍然需要确认
    rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(img.device)
    img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                       rgb_weights) / 255. + 16.
    return img


class TargetNorm:
    """
    input: [0, 1]
    output:[-1, 1]
    """
    def __call__(self, img):
        img = 2. * img - 1
        return img


class ScalableCrop:
    """
    Need pil input
    crop the largest image whose size can be divided by scaling_factor
    """
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, img):
        h_scaled = int(img.height / self.scaling_factor)
        w_scaled = int(img.width / self.scaling_factor)
        crop = transforms.CenterCrop([h_scaled * self.scaling_factor, w_scaled * self.scaling_factor])
        img = crop(img)
        return img


# class DownSampleResize:
#     # 有莫名其妙的彩色噪点, 已解决
#     # antialias
#     """
#     Need Tensor input
#     down sample by scaling_factor, using interpolation
#     """
#     def __init__(self, scaling_factor, interpolation):
#         self.scaling_factor = scaling_factor
#         self.interpolation = interpolation
#
#     def __call__(self, img):
#         h_scaled = int(img.shape[-2] / self.scaling_factor)
#         w_scaled = int(img.shape[-1] / self.scaling_factor)
#         resize = transforms.Resize(
#             [h_scaled, w_scaled],
#             interpolation=self.interpolation,
#             antialias=True  # antialias
#         )
#         # img = resize(img)
#         img = resize(img).clamp(min=0, max=1)
#         return img

class DownSampleResize:
    """
    Need pil input
    down sample by scaling_factor, using interpolation
    """
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, img):
        h_scaled = int(img.height / self.scaling_factor)
        w_scaled = int(img.width / self.scaling_factor)
        rs_img = img.resize((w_scaled, h_scaled), Image.Resampling.BICUBIC)
        return rs_img
