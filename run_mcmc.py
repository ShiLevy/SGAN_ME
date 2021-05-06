# -*- coding: utf-8 -*-
"""
A Pytorch implementation in Python 3.6 of the GAN-based probabilistic inversion 
approach by Laloy et al. (2018a). The inversion code is the DREAMzs (ter Braak and Vrugt, 2008; 
Vrugt, 2009; Laloy and Vrugt, 2012) MCMC sampler and the considered toy forward 
problem involves 2D GPR linear tomography in a binary channelized subsurface domain.

@author: Eric Laloy <elaloy@sckcen.be>

Please drop me an email if you have any question and/or if you find a bug in this
program. 

===
Copyright (C) 2018  Eric Laloy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ===                               

References:
    
Laloy, E., Hérault, R., Jacques, D., and Linde, N. 2018a. Training-image based 
    geostatistical inversion using a spatial generative adversarial neural network. 
    Water Resources Research, 54, 381–406. https://doi.org/10.1002/2017WR022148.

Laloy, E., Linde, N., Ruffino, C., Hérault, R., & Jacques, D. 2018b. Gradient-based 
    deterministic inversion of geophysical data with Generative Adversarial Networks: 
    is it feasible? arXiv:1812.09140, https://arxiv.org/abs/1812.09140.                                     
                                                                                                                                                                                                       
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

work_dir=os.getcwd()

import mcmc

#% Set random seed and case study

rng_seed=12345

CaseStudy = 2
percentile = '70b'


#---------------------------------------------------------------------------------------------------------------------------------
# delayed model error scenario. model error is switched from 0 to 1
delayed_err = 0
stage = 5000
#---------------------------------------------------------------------------------------------------------------------------------
# If to gradually introduce model error contribution, 0-no, 1-yes
weighted_err = 0
#---------------------------------------------------------------------------------------------------------------------------------
#invert a constant to scale model errors
err_scale=1 
#---------------------------------------------------------------------------------------------------------------------------------
#Tempering scenario
tempering = 1       # to tune jr_scale automatically, set to 1 with T=1
minJRS = 0.2
T = 1           #Temperature for annealing
burnin = 20000   #end iteration for burn-in period (stop tempering)

if  CaseStudy==0: #100-d correlated gaussian (case study 2 in DREAMzs Matlab code)
    seq=3
    steps=5000
    ndraw=seq*100000
    thin=10
    jr_scale=1.0
    Prior='LHS'

if  CaseStudy==1: #10-d bimodal distribution (case study 3 in DREAMzs Matlab code)
    seq=5
    ndraw=seq*40000
    thin=10
    steps=np.int32(ndraw/(20.0*seq))
    jr_scale=1.0
    Prior='COV'
    
if  CaseStudy==2: # Linear GPR tomography (toy) problem: accounting for model errors using WGAN generator
    seq=8
    ndraw=seq*100000
    thin=50              # how often to save the output: Sequences
    steps=10           # how often to calculate the diagnostics
    jr_scale=5       # bigger jumps --> bigger
    Prior='LHS'         # not completely random
    residual = 0
    
if CaseStudy==3: # Linear GPR tomography (toy) problem: accounting for model errors using covariance matrix (Hansen et al. 2014)
    seq=8
    ndraw=seq*100000
    thin=50
    steps=10
    jr_scale=5
    Prior='LHS'

if CaseStudy==4: # Linear GPR tomography (toy) problem: not accounting for model errors
    seq=8
    ndraw=seq*10000
    thin=50
    steps=10
    jr_scale=5
    Prior='LHS'
    #add choice of log liklihood for each case
    
if CaseStudy==5: # inverting the model error
    seq=8
    ndraw=seq*10000
    thin=10                  # determine how often to save samples to sequences (*not* to X and Z!)
    steps=10                # determine how often to calculate diagnostics: AR (average over those number of steps) and GR
    jr_scale=5
    residual = 0            # if the error was trained with mean-std-normalized samples (1) otherwise (0)
    Prior='LHS'
    #add choice of log liklihood for each case
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DoParallel=True # forward model is so quick here that parallel is not necessary
parallel_jobs=None

if not (CaseStudy==2 or CaseStudy==5):
    residual = 0

scen = [delayed_err,weighted_err,err_scale]   
if np.sum(scen)>1:
    print('#### Error: you have more than one case for dealing with model errors on, make sure that only one is set to 1 ####')
    sys.exit()

#% Run the DREAMzs algorithm
if __name__ == '__main__':
    
    start_time = time.time()

    q=mcmc.Sampler(main_dir=work_dir,CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,Prior=Prior,parallel_jobs=seq,steps=steps,
                   parallelUpdate =0.8 ,pCR=True,thin=thin,nCR=3,DEpairs=1,pJumpRate_one=0.2,BoundHandling='Fold',
                   lik_sigma_est=False,DoParallel=DoParallel,jr_scale=jr_scale,rng_seed=rng_seed,device=True,
                   saveout=True,save_tmp_out=True,res_calc=residual,percentile=percentile, tempering=tempering,
                   T=T, burnin=burnin,weighted_err=weighted_err,delayed_err=delayed_err,stage=stage,err_scale=err_scale,minJRS=minJRS)
    
    print("Iterating")
    
    #tmpFilePath = None
    tmpFilePath = './'+str(percentile)+'percentile/c'+str(CaseStudy)+'/out_tmp.pkl' # None or: work_dir+'\out_tmp.pkl' for a restart
    
    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar, Extra, DNN = q.sample(RestartFilePath=tmpFilePath)
    
    OutDiag, MCMCPar, MCMCVar, Extra, DNN = dict(OutDiag), dict(MCMCPar), dict(MCMCVar), dict(Extra), dict(DNN)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time))
    
    # if plot_results:
        
# pCR: for the first iterations if set to True it will adapt the nCR if false, the nCR will be the one set in nCR variable.
# fx: last time travels 
