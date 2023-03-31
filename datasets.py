# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.12 14:27
# @Author  : James.T
# @File    : dataset.py

import os
import sys
import json
import utils
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms
import torchvision


img_list_file_name = f'dataset_image_list.json'


class DatasetFromFolder(Dataset):
    """
    Basic SR Dataset
    Return HR and LR
    """
    def __init__(
            self,
            root: str,
            transform_prep: Callable,
            transform_hr: Optional[Callable] = None,
            transform_lr: Optional[Callable] = None,
    ):
        """
        initialization

        :param root: dataset folder
        :param transform_prep: pre process
        :param transform_hr: transforms applied to get high resolution image
        :param transform_lr: transforms applied to get low resolution image
        :returns: None
        """
        self.data_folder = root  # 需要具体指定某个数据集的文件夹
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.transform_prep = transform_prep

        if not os.path.exists(self.data_folder):
            print('Dataset: data folder no exists')
            sys.exit()

        with open(os.path.join(self.data_folder, img_list_file_name), 'r') as jsonfile:
            self.image_list = json.load(jsonfile)

    def __getitem__(self, idx):
        """
        get one image
        """
        img = Image.open(self.image_list[idx], mode='r')
        img = img.convert('RGB')
        img = self.transform_prep(img)
        if self.transform_hr and self.transform_lr:
            hr_img = self.transform_hr(img)
            lr_img = self.transform_lr(img)
            return lr_img, hr_img
        else:
            return img

    def __len__(self):
        """
        the quantity of images
        """
        return len(self.image_list)


def SRDataset(
        dataset_name: str,
        lr_target_type: Optional[str] = None,
        hr_target_type: Optional[str] = None,
        crop_size: Optional[int] = None,
        scaling_factor: Optional[int] = None,
) -> DatasetFromFolder:
    """
    build a dataset

    :param dataset_name:
        the name of the dataset, according to the name, get the path
    :param lr_target_type:

    :param hr_target_type:

    :param crop_size:
        determine the dataset is used fot training or not.
        when True, the output HR and LR image will be cropped
        the output HR image size
    :param scaling_factor:
        down sample HR image by scaling_factor

    :returns:
        the configured dataset
    """
    # 数据集名字 到 路径的 dict 映射，检查数据集是否存在
    datasets_dict = {
        'Set5': './data/Set5',
        'Set14': './data/Set14',
        'BSD100': './data/BSD100',
        'COCO2014': './data/COCO2014',
        'DF2K_OST': './data/DF2K_OST',
    }
    assert dataset_name in datasets_dict.keys(), f'{dataset_name} doesnt exist'
    dataset_path = datasets_dict[dataset_name]

    # img_type
    img_type_list = ['imagenet-norm', '[-1, 1]']
    assert lr_target_type in img_type_list, f'no image type named {lr_target_type}'
    assert hr_target_type in img_type_list, f'no image type named {hr_target_type}'

    # 检查 crop 后的图片是否能被整除下采样
    is_crop = bool(crop_size)
    if is_crop:
        assert crop_size % scaling_factor == 0, '剪裁尺寸不能被放大比整除'

    # 检查 json 是否存在，没有就创建
    if not os.path.isfile(os.path.join(dataset_path, img_list_file_name)):
        print('Dataset: image_list doesnt exist, creating...')
        create_data_list(dataset_path, crop_size)

    # 选择 crop 的形式
    if is_crop:
        prep = transforms.RandomCrop(crop_size)
    else:
        prep = utils.ScalableCrop(scaling_factor=scaling_factor)

    # lr 变换
    if lr_target_type:
        trans_lr = utils.compose_lr_transforms(
            img_type=lr_target_type,
            scaling_factor=scaling_factor
        )
    else:
        trans_lr = None

    # hr 变换
    if hr_target_type:
        trans_hr = utils.compose_hr_transforms(
            img_type=hr_target_type
        )
    else:
        trans_hr = None

    dataset = DatasetFromFolder(
        root=dataset_path,
        transform_prep=prep,
        transform_hr=trans_hr,
        transform_lr=trans_lr,
    )
    return dataset


def create_data_list(data_folder: str, min_size: int):
    """
    make a json-style list file which consists all image paths in data_folder.
    and filter images smaller than min_size when is_train is True.

    :param data_folder: dataset folder
    :param min_size: the smallest acceptable image size
    :param is_train: When True, filter min_size

    :returns: None
    """
    image_list = []
    is_crop = bool(min_size)
    for root, dirs, files in os.walk(data_folder): # 可用 遍历 栈实现，但是用 os.walk()
        for name in files:
            _, filetype = os.path.splitext(name)
            if not filetype.lower() in ['.jpg', '.bmp', '.png']:
                continue
            img_path = os.path.join(root, name)
            if is_crop:
                img = Image.open(img_path, mode='r')
                if not (img.width >= min_size and img.height >= min_size):
                    continue
            image_list.append(img_path)
    with open(os.path.join(data_folder, img_list_file_name), 'w') as jsonfile:
        json.dump(image_list, jsonfile)


# dataset folder arrangement
# dataset/
#   -> .../
#   -> dataset_image_list.json


if __name__ == '__main__':

    # test

    td = SRDataset('COCO2014', crop_size=96, scaling_factor=4)
    # print(td.image_list)
    # print(len(td))
    # imgs = [td[i] for i in range(len(td))]

    # # test transforms
    # lr, hr = td[0]
    # utils.show_pil_image(hr)
    # img_ = utils.ImageTransforms('test', 0, 4, 'imagenet-norm', 'imagenet-norm')(hr)
    # utils.show_tensor_image(img_)
    # img_my = utils.compose_transforms(4, is_train=False)[0](hr)
    # utils.show_tensor_image(img_my)
    # utils.show_tensor_image(img_my - img_)

    # lr, hr = td[0]
    # utils.show_tensor_image(hr)

    lr, hr = td[0]
    utils.show_tensor_image(hr)
    utils.show_tensor_image(lr)

    # # two ways to denormalize img
    # trans = utils.compose_denormalize_transforms()
    # hr_t = trans(hr)
    # utils.show_tensor_image(hr_t)
    #
    # grid = torchvision.utils.make_grid(hr, normalize=True)
    # utils.show_tensor_image(grid)
