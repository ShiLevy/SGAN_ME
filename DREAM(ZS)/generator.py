"""This is a copy of the SGAN2D_G generator network found in nnmodels.py, but with a
few modifications to make it easier to use after it has been trained. File mostly taken
from the companion code of Richardson (Arxiv June 2018)
@author: Eric Laloy <elaloy elaloy@sckcen.be>
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, gpath, cuda=True, nc = 1, nz = 8, ngf = 64, gfs = 5):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
        
            nn.ConvTranspose2d(     nz, ngf * 8, gfs, 2, 3, bias=False), #stride=1, pad=0 
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf * 8),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, 2, 3, bias=False),#stride=2, pad=1
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf * 4),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, 2, 3, bias=False),#stride=2, pad=1
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf * 2),
           
            nn.ConvTranspose2d(ngf * 2,     ngf, gfs, 2, 3, bias=False),#stride=2, pad=1
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf),
            
            nn.ConvTranspose2d(    ngf,      nc, gfs, 2, 4, bias=False),#stride=2, pad=1
            nn.Tanh()
            
        )
        if cuda:
            self.load_state_dict(torch.load(gpath))
        else:
            self.load_state_dict(torch.load(gpath,
                                            map_location=lambda storage,
                                            loc: storage))

    def forward(self, input):
        output = self.main(input)
        return output
    