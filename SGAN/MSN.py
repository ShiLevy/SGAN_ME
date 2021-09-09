"""
This piece of code was taken from https://github.com/AntixK/mean-spectral-norm/blob/master/SNlayers.py repository. It applies speactral normalization based on the paper "Mean Spectral Normalization of Deep Neural Networks for Embedded Automation" [ArXiv Version] by Anand Krishnamoorthy Subramanian and Nak-Young Chong, 2019.

The MIT License (MIT)

Copyright (c) 2019

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

"""


import torch
import torch.nn as nn

class MeanSpectralNorm(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0,1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        #sigma2 = y.var(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
            #y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                    #self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
            y = y - mu.view(-1,1)
            #y = y / (sigma2.view(-1,1)**.5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0,1)
