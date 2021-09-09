# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@ original author: Eric Laloy <elaloy@sckcen.be> (original code available here:
https://github.com/elaloy/SGANinv/tree/master/braided_river_pytorch)
    
@ modifications by Shiran Levy <shiran.levy@unil.ch> June 2021
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from collections import OrderedDict
from MSN import MeanSpectralNorm


class SGAN2D_Dwass(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dfs = 9, ngpu = 1, s = 2, p = 2):
        super(SGAN2D_Dwass, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(

            nn.Conv2d(nc, ndf, dfs, s, dfs//2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf),

            nn.Conv2d(ndf, ndf*2, dfs, s, dfs//2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 2),

            nn.Conv2d(ndf*2, ndf*4, dfs, s, dfs//2, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 4),

            nn.Conv2d(ndf*4, ndf*8, dfs, s, dfs//2, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 8),

            nn.Conv2d(ndf * 8, 1, 1, 2, bias=False)
            
        )
        self.main = main
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean()
        
        return output.view(1)

class SGAN2D_D_SNorm(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dfs = 9, ngpu = 1, s = 2, p = 2):
        super(SGAN2D_D_SNorm, self).__init__()
        self.ngpu = ngpu
        
        main = nn.Sequential(
            
            #performs spectral normalization on the weights, replacing weight clipping 
            sn(nn.Conv2d(nc, ndf, dfs, s, dfs//2, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn(nn.Conv2d(ndf, ndf*2, dfs, s, dfs//2, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf*2, ndf*4, dfs, s, dfs//2, bias=False)),  
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf*4, ndf*8, dfs, s, dfs//2, bias=False)), 
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf * 8, 1, 1, 2, bias=False)),
        )
        self.main = main

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean()

        return output.view(1)

class SGAN2D_D_MSN(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dfs = 9, ngpu = 1, s = 2, p = 2):
        super(SGAN2D_D_MSN, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(

            #performs mean spectral normalization on the weights, replacing weight clipping 
            sn(nn.Conv2d(nc, ndf, dfs, s, dfs//2,  bias=False)),
            MeanSpectralNorm(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf, ndf*2, dfs, s, dfs//2, bias=False)),
            MeanSpectralNorm(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf*2, ndf*4, dfs, s, dfs//2, bias=False)),
            MeanSpectralNorm(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf*4, ndf*8, dfs, s, dfs//2, bias=False)),
            MeanSpectralNorm(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf * 8, 1, 1, 2, bias=False)),
        )
        self.main = main

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean()

        return output.view(1)
    

class SGAN2D_G(nn.Module):
    def __init__(self, nc = 1, nz = 1, ngf = 64, gfs = 5, ngpu = 1, s = 2, p = 2, AF = 'relu'):
        super(SGAN2D_G, self).__init__()
        self.ngpu = ngpu

        if AF == 'prelu':
            activation = nn.PReLU(nc, init = 0.2)
        else:
            activation = nn.ReLU(True)

        self.main = nn.Sequential(

                nn.ConvTranspose2d(     nz, ngf * 8, gfs, s, p, bias=False),
                activation,
                nn.InstanceNorm2d(ngf * 8),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, s, p, bias=False),
                activation,
                nn.InstanceNorm2d(ngf * 4),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, s, p, bias=False),
                activation,
                nn.InstanceNorm2d(ngf * 2),

                nn.ConvTranspose2d(ngf * 2,     ngf, gfs, s, p, bias=False),
                activation,
                nn.InstanceNorm2d(ngf),

                nn.ConvTranspose2d(    ngf,      nc, gfs, s, 4, bias=False),
                nn.Tanh()

            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output

