'''
训练分为两个主要部分，第1部分更新判别器，第2部分更新生成器。

1 训练判别器
   a. 从训练集构建一批实际样本，前向通过D,计算损失logD(x),然后计算后向梯度
   b. 用当前生成器构造一批假样本，通过D前向传递该batch,计算损失log(1-D(G(z)))
      并通过后向传播传递梯度
2 训练生成器
   最小化log(1-D(G(z)))来训练生成器，以便产生更好的伪样本，但是这样不会提供足够的梯度（Goodfellow),
   作为修复希望最大化D(G(z)),
   通过以下方式实现（trick)：
   使用D对step1中的G中的输出进行分类，使用真实标签(label=1)计算G的损失，后向传播计算G的梯度，最后使用优化器逐步
   更新G的参数

最后，在每个epoch结束时，经通过生成器G推送fixed_noise batch,用来直观的跟踪G训练的进度
Los_D = log(D(x)) + log(1-D(G(x)))
Loss_G = log(D(G(x)))
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
import matplotlib
matplotlib.use('Agg')

from config import *
from model import *

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
save_fig_index = 0


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epoch):
    # 对于数据加载器中的每个batch
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epoch-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_tensor_i = vutils.make_grid(fake, padding=2, normalize=True)
            img_list.append(img_tensor_i)
            
            vutils.save_image(img_tensor_i, "./fake_g/"+str(save_fig_index)+".jpg")
            save_fig_index += 1

        iters += 1

# plot loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
# plt.show()

# real vs fake generator

real_batch = next(iter(dataloader))

# 绘制真实图像
vutils.save_image(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),"real.jpg")
vutils.save_image(vutils.make_grid(img_tensor_i, padding=5, normalize=True).cpu(),"fake.jpg")




