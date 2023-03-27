import numpy as np

import utils
from PIL import Image
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import random
import torchvision.transforms.functional as FT
import torch
import math

# loss = utils.load_loss_file('epoch_loss_0_129.npy')
# utils.draw_loss_curve(loss)

class DownSampleResize:
    # 有莫名其妙的彩色噪点
    # antialias
    """
    Need Tensor input
    down sample by scaling_factor, using interpolation
    """
    def __init__(self, scaling_factor, interpolation):
        self.scaling_factor = scaling_factor
        self.interpolation = interpolation

    def __call__(self, img):
        h_scaled = int(img.shape[-2] / self.scaling_factor)
        w_scaled = int(img.shape[-1] / self.scaling_factor)
        resize = transforms.Resize(
            [h_scaled, w_scaled],
            interpolation=self.interpolation,
            antialias=True  # antialias
        )
        img = resize(img)
        return img



device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    图像变换.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """

        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # 下采样（双三次差值）
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # 安全性检查
        # assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


# img = Image.open('./data/Set5/bird_GT.bmp', mode='r')
img = Image.open(r'E:\Resourses\Photograph\Wallpaper\seel.jpg', mode='r')
img.convert('RGB')

mytrans1 = transforms.Compose(
    [
        transforms.ToTensor(),
        utils.ScalableCrop(scaling_factor=4),
    ])

#mytrans2 = utils.compose_hr_transforms()
mytrans_lr = DownSampleResize(
                4,
                transforms.InterpolationMode.BICUBIC
            )
mytrans_hr = utils.compose_hr_transforms()
#myimg = mytrans2(mytrans1(img))
myimg_hr = mytrans1(img)
myimg_lr = mytrans_lr(myimg_hr)
# myimg_hr = mytrans_hr(myimg_hr)

trans = ImageTransforms(
    split='test',
    crop_size=1,
    scaling_factor=4,
    lr_img_type='[0, 1]',
    hr_img_type='[0, 1]',
)

imgt_lr, imgt_hr = trans(img)

imgt_np = imgt_lr.numpy().reshape((-1))
print(imgt_np.dtype)
myimg_np = myimg_lr.numpy().reshape((-1))
print(myimg_np.dtype)

# histo = np.concatenate((imgt_np, myimg_np), axis=1)


plt.figure()
plt.hist(x = (imgt_np, myimg_np),            # 绘图数据
        bins = 255,            # 指定直方图的条形数为20个
        edgecolor = 'w',      # 指定直方图的边框色
        color = ['c','r'],    # 指定直方图的填充色
        label = ['第一组','第二组'],     # 为直方图呈现图例
        density = False,      # 是否将纵轴设置为密度，即频率
        alpha = 0.6,          # 透明度
        rwidth = 1,           # 直方图宽度百分比：0-1
        stacked = False)      # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
plt.legend()

# utils.show_tensor_image(myimg)
# utils.show_tensor_image(imgt)
# utils.show_tensor_image(myimg - imgt)
utils.show_tensor_image(myimg_lr)
utils.show_tensor_image(imgt_lr)
utils.show_tensor_image(myimg_lr - imgt_lr)
utils.show_tensor_image(myimg_hr)
utils.show_tensor_image(imgt_hr)
utils.show_tensor_image(myimg_hr - imgt_hr)
plt.show()
