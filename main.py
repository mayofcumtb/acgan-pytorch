"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from syn_data.syn_dataset import SynImageDatasets
from torchvision.

import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist') # 数据集名称
parser.add_argument('--dataroot', required=True, help='path to dataset') # 数据集路径
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2) # 启用多线程进行数据读取
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector') # 随机向量维度
parser.add_argument('--ngf', type=int, default=64) # number of generator filers
parser.add_argument('--ndf', type=int, default=64) # number of discriminater filters
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for') # 遍历几遍代码
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')# adam 优化器的参数
parser.add_argument('--cuda', action='store_true', help='enables cuda') # 是否使用gpu进行加速训练
parser.add_argument('--netG', default='', help="path to netG (to continue training)") # netG的模型参数保存位置
parser.add_argument('--netD', default='', help="path to netD (to continue training)") # netD的模型参数保存位置
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints') #模型的参数保存位置
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])
    )

elif opt.dataset == 'syn':
    dataset= SynImageDatasets(root=opt.dataroot, transform=transforms.Scale(opt.imageSize))
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
if opt.dataset == 'mnist':
    nc = 1
    nb_label = 10
elif opt.dataset == 'cifar10':
    nc = 3
    nb_label = 10
else:
    nc = 3
    nb_label = 12

netG = model.netG(nz, ngf, nc)

if opt.netG != '':
    print ("load generator net params from %s"%opt.netG)
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.netD(ndf, nc, nb_label)

if opt.netD != '':
    print ("load discriminator net params from %s"%opt.netD)
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

s_criterion = nn.BCELoss()# which source is real or synthesis
c_criterion = nn.NLLLoss()
#TODO: last thing compare
"""
add patch discrimnator loss
"""

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
s_label = torch.FloatTensor(opt.batchSize)
c_label = torch.LongTensor(opt.batchSize)

real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
random_label = np.random.randint(0, nb_label, opt.batchSize)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((opt.batchSize, nb_label))
random_onehot[np.arange(opt.batchSize), random_label] = 1
fixed_noise_[np.arange(opt.batchSize), :nb_label] = random_onehot[np.arange(opt.batchSize)]


fixed_noise_ = (torch.from_numpy(fixed_noise_).clone())
fixed_noise_ = fixed_noise_.resize_(opt.batchSize, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

def add_background_to_syn_image(back, syn_image, alpha):
    import pdb; pdb.set_trace()
    for i in range(len(back)):
        back[i]


for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        netD.zero_grad()
        syn_image, real_image, syn_alpha, syn_class_index, real_class_index, azimuth, elevation, tilt, distance = data
        batch_size = real_image.size(0)
        input.data.resize_(real_image.size()).copy_(real_image)
        s_label.data.resize_(batch_size).fill_(real_label)
        c_label.data.resize_(batch_size).copy_(real_class_index)
        s_output, c_output = netD(input)
        s_errD_real = s_criterion(s_output, s_label)
        c_errD_real = c_criterion(c_output, c_label)
        errD_real = s_errD_real + c_errD_real
        errD_real.backward()
        D_x = s_output.data.mean()
        
        correct, length = test(c_output, c_label)

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        label = syn_class_index
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label, azimuth, elevation, tilt, distance] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
        
        noise_ = (torch.from_numpy(noise_).clone())
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.data.copy_(noise_)

        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        add_background_to_syn_image(fake, syn_image, syn_alpha)
        """
        need to add a function to pass the background and syn_obj with alpha
        """
        s_label.data.fill_(fake_label)
        s_output,c_output = netD(fake.detach())
        s_errD_fake = s_criterion(s_output, s_label)
        c_errD_fake = c_criterion(c_output, c_label)
        errD_fake = s_errD_fake + c_errD_fake

        errD_fake.backward()
        D_G_z1 = s_output.data.mean()
        errD = s_errD_real + s_errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        s_label.data.fill_(real_label)  # fake labels are real for generator cost
        s_output,c_output = netD(fake)
        s_errG = s_criterion(s_output, s_label)
        c_errG = c_criterion(c_output, c_label)
        
        errG = s_errG + c_errG
        errG.backward()
        D_G_z2 = s_output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2,
                 correct, length, 100.* correct / length))
        if i % 100 == 0:
            vutils.save_image(img,
                    '%s/real_samples.png' % opt.outf)
            #fake = netG(fixed_cat)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
