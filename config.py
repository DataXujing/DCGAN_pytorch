'''

data: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML

# 设置随机数种子
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 如果你想要新的结果就是要这段代码
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 数据集根目录
# dataroot = "data/celeba"
dataroot = "data"


# 加载数据的工作线程数
workers = 2

# batch size
batch_size = 128

# 训练图像空间大下
image_size = 64

# 训练图像的通道数
nc = 3

# 潜变量 Z的大下（生成器输入的大小)
nz = 100

# 生成器中特征图的大小
ngf = 64

# 判别器中特征映射的大小

ndf = 64

# epoch
num_epoch = 100

# lr
lr = 0.0002

# optimizor Adam
beta1 = 0.5

# nym gpus
ngpu = 1


# 我们可以按照设置的方式使用图像文件夹数据集。
# 用ImageFolder数据集类，它要求在数据集的根文件夹中有子目录
# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 选择我们运行在上面的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



if __name__ == "__main__":
    # 绘制部分我们的输入图像
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
