# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@ original author: Eric Laloy <elaloy@sckcen.be> (original code available here:
https://github.com/elaloy/SGANinv/tree/master/braided_river_pytorch)
    
@ modifications by Shiran Levy <shiran.levy@unil.ch> June 2021

"""
from __future__ import print_function
import argparse
import os
import sys
from tqdm import tqdm
from time import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchsummary import summary
from nnmodels import SGAN2D_G
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import platform
import time
from utils import zx_to_npx, det_err_min_max, get_err_iter_int

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=1, help='number of non-spatial dimensions in latent space z')
parser.add_argument('--zx', type=int, default=5, help='number of grid elements in x-spatial dimension of z')
parser.add_argument('--zy', type=int, default=5, help='number of grid elements in x-spatial dimension of z')
parser.add_argument('--nc', type=int, default=1, help='number of channeles in original image space')
parser.add_argument('--ngf', type=int, default=64, help='initial number of filters for dis')
parser.add_argument('--ndf', type=int, default=64, help='initial number of filters for gen')
parser.add_argument('--dfs', type=int, default=5, help='kernel size for dis')
parser.add_argument('--gfs', type=int, default=5, help='kernel size for gen')
parser.add_argument('--strideD', type=int, default=2, help='stride for for dis')
parser.add_argument('--strideG', type=int, default=2, help='stride for for gen')
parser.add_argument('--Pad', type=int, default=3, help='Padding for for dis&gen')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--niter', type=int, default=500, help='number of iterations per training epoch')
parser.add_argument('--lr', type=float, default=0.000001, help='learning rate, default=0.0002')
#parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--critic', default='SN', help='type of GAN, WC-weight clipping, SN- spectral normalization, GAN- classical gan')
parser.add_argument('--AF', default='relu', help='type of activation func. in the gen.: relu or prelu')
parser.add_argument('--manualSeed', type=int, default=1978,help='manual seed')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--Restart', type=int, default=0, help='Restart training')
parser.add_argument('--z_dist', default='uniform', help='Z-space distribution')
parser.add_argument('--optimize', default='RMSProp', help='type of optimizer RMSProp or Adam')
opt = parser.parse_args()
print(opt)
outf = './Model_error_save' #name of new created folder to save checkpoints into

try:
    os.makedirs(outf)
except OSError:
    pass
print("Random Seed: ", opt.manualSeed)

real_stat = torch.zeros(2,int(opt.nepoch))		#initializing ststistics vector

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
np.random.seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
device = torch.device("cuda:0" if opt.cuda else "cpu")

if platform.system()=='Windows':
    
    torch.backends.cudnn.enabled = True
    
ngpu = int(opt.ngpu)
strideG = int(opt.strideG)
strideD = int(opt.strideD)
pad = int(opt.Pad)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
dfs = int(opt.dfs)
gfs = int(opt.gfs)
nc = int(opt.nc)
zx = int(opt.zx)
zy= int(opt.zy)
zx_sample = zx
zy_sample = zy
depth=5
npx=zx_to_npx(zx,depth,dfs,strideG,pad)
npy=zx_to_npx(zy,depth,dfs,strideG,pad)

batch_size = int(opt.batchSize)

print(npx,npy)

folder_err = "./database/" #directory of database
scalefac, extrema = det_err_min_max(folder_err) 
del extrema
data_iter = get_err_iter_int(folder_err, scalefac, npx, batch_size=batch_size, n_channel=nc)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Wasserstein loss with clipping
if opt.critic == 'WC':	
    #Wasserstein GAN
    from nnmodels import SGAN2D_Dwass as SGAN2D_D
    clamp_lower = -0.01
    clamp_upper = 0.01
    opt.Diters = 5

#Wasserstein loss with spectral normalization
elif opt.critic == 'SN':	
    from nnmodels import SGAN2D_D_SNorm as SGAN2D_D    
         
#Wasserstein loss with Mean spectral normalization
elif opt.critic == 'MSN':
    from nnmodels import SGAN2D_D_MSN as SGAN2D_D
        
netG = SGAN2D_G(nc, nz, ngf, gfs, ngpu, strideG, pad, opt.AF)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = SGAN2D_D(nc, ndf, dfs, ngpu = 1, s = strideD, p = pad)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Optimizers
if opt.optimize == 'RMSProp':
    optimizerD = optim.RMSprop(filter(lambda p: p.requires_grad, netD.parameters()), lr=4*opt.lr, momentum = 0)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr, momentum = 0)
elif opt.optimize == 'Adam':
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

input_noise = torch.rand(batch_size, nz, zy, zx, device=device)*2-1
fixed_noise = torch.rand(500, nz, zy_sample, zx_sample, device=device)*2-1

input = torch.FloatTensor(batch_size, nc, npy, npx)

one = torch.FloatTensor([1]).cuda()
mone = one * -1
cur_ep = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    input_noise, fixed_noise = input_noise.cuda(), fixed_noise.cuda()

Gloss, Closs= [], []
gen_iterations = 0

if opt.Restart:
    print('Restarting from last epoch')
    state = torch.load(outf+'/RestartFile.t7',map_location = device)
    netG.load_state_dict(state['netG'])
    netD.load_state_dict(state['netD'])
    optimizerG.load_state_dict(state['optimizerG'])
    optimizerD.load_state_dict(state['optimizerD'])
    fixed_noise = state['fixed_noise']
    cur_ep = state['epoch']
    Gloss = state['Gloss']
    Closs = state['Closs']
    gen_iterations = state['gen_iterations']
    del state

for epoch in range(cur_ep+1 if opt.Restart else 0, opt.nepoch):
    i=0
    while i < opt.niter:

        ############################
        # (1) Update D
        ###########################
        
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < opt.niter:
            j += 1
 
            if opt.critic == 'WC':
                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

            data = next(data_iter)
            
            i += 1
            
            # train with real
            netD.zero_grad()

            real_cpu = torch.Tensor(data).to(device)
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv)
            errD_real.backward(one)

             # train with fake
            if opt.z_dist == 'normal':    #normal distribution N(0,1)
                noise = torch.randn(batch_size, nz, zy, zx, device=device)
            else:			  #uniform distribution [-1,1]
                noise = torch.rand(batch_size, nz, zy, zx, device=device)*2-1

            with torch.no_grad():
                noisev=Variable(noise)

            fake = Variable(netG(noisev).data) # freezes the weights of netG, so that the weights are not updated for the generator
            inputv = fake
            errD_fake = netD(inputv)
            errD_fake.backward(mone)

            errD = errD_real - errD_fake
            optimizerD.step()
	
        Closs.append(errD.detach().cpu().numpy())
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()

        if opt.z_dist == 'normal':
            noise = torch.randn(batch_size, nz, zy, zx, device=device)
        else:
            noise = torch.rand(batch_size, nz, zy, zx, device=device)*2-1
            
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)
        
        errG.backward(one)
        Gloss.append(-errG.detach().cpu().numpy())
        optimizerG.step()
        gen_iterations += 1
        
        print(i)
        print(gen_iterations)
        
        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(data), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        
    # do checkpointing
    print('epoch ',epoch,' done')
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

    fake = netG(fixed_noise)
    vutils.save_image(torch.mean(fake.detach(),dim=0),
            '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
            normalize=True)

#save a file at the end of training for restart
state = {
'epoch': epoch,
'netG': netG.state_dict(),
'netD': netD.state_dict(),
'optimizerG': optimizerG.state_dict(),
'optimizerD': optimizerD.state_dict(),
'fixed_noise': fixed_noise,
'Gloss': Gloss,
'Closs': Closs,
'gen_iterations': gen_iterations}
torch.save(state, outf+'/RestartFile.t7')

