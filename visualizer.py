import numpy as np
import utils
from PIL import Image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import math

fig_size = (10, 4)


# ------------------------------------------------------------------------------------------
# SRResNet MSE epoch
loss_file = r'.\task_record\20230325_0130_Saturday_test_env\record\epoch_loss_0_129.npy'
loss = utils.load_loss_file(loss_file)
w = 20
aver_loss = np.convolve(loss, np.ones(w), 'same') / w

plt.figure(figsize=fig_size)
plt.plot(loss, alpha=0.2, label='loss')
plt.plot(aver_loss, label='loss_20', color='orange', linewidth=2.5)

plt.ylim((0.015, 0.02))
plt.xlim((10, 120))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tick_params(tickdir='in')

plt.grid(axis='y')

plt.title('SRResNet MSE Loss')

plt.legend(loc='best', edgecolor='black')
plt.show()


# ------------------------------------------------------------------------------------------
# SRResNet MSE iter
loss_file = r'.\task_record\20230325_0130_Saturday_test_env\record\iter_loss_0_129.npy'
loss = utils.load_loss_file(loss_file)
w = 850
aver_loss = np.convolve(loss, np.ones(w), 'same') / w

plt.figure(figsize=fig_size)
plt.plot(loss, alpha=1, color='orange', label='loss')

# plt.ylim((0.015, 0.02))
# plt.xlim((10, 120))
plt.xlabel('iter.')
plt.ylabel('loss')
plt.tick_params(tickdir='in')

plt.grid(axis='y')

plt.title('SRResNet MSE Loss')

plt.legend(loc='best', edgecolor='black')
plt.show()

# ------------------------------------------------------------------------------------------
# SRGAN discriminator epoch
loss_file_d = r'.\task_record\20230325_2327_Saturday_test_srgan\record\loss_epochs_d_0_79.npy'
loss_d = utils.load_loss_file(loss_file_d)
loss_file_a = r'.\task_record\20230325_2327_Saturday_test_srgan\record\loss_epochs_g_adversarial_0_79.npy'
loss_a = utils.load_loss_file(loss_file_a)
loss_file_c = r'.\task_record\20230325_2327_Saturday_test_srgan\record\loss_epochs_g_content_0_79.npy'
loss_c = utils.load_loss_file(loss_file_c)

w = 10
aver_loss_d = np.convolve(loss_d, np.ones(w), 'same') / w
aver_loss_a = np.convolve(loss_a, np.ones(w), 'same') / w
aver_loss_c = np.convolve(loss_c, np.ones(w), 'same') / w

plt.figure(figsize=fig_size)

plt.subplot(1, 3, 1)
plt.plot(aver_loss_d, color='lime', linewidth=2.5)
plt.title('discriminator')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tick_params(tickdir='in')
plt.grid(axis='y')
plt.xlim((5, 75))

plt.subplot(1, 3, 2)
plt.plot(aver_loss_a, color='royalblue', linewidth=2.5)
plt.title('generator adversarial')
plt.xlabel('epoch')
plt.tick_params(tickdir='in')
plt.grid(axis='y')
plt.xlim((5, 75))

plt.subplot(1, 3, 3)
plt.plot(aver_loss_c, color='fuchsia', linewidth=2.5)
plt.title('generator content')
plt.xlabel('epoch')
plt.tick_params(tickdir='in')
plt.grid(axis='y')
plt.xlim((5, 75))

plt.show()

