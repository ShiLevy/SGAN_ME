# -*- coding: utf-8 -*-
"""
A Pytorch implementation in Python 3.6 of the GAN-based probabilistic inversion 
approach by Laloy et al. (2018a). The code was extended to account for model errors using the 
SGAN (SGAN-ME; Levy et al., under review) or by inflating the covariance matrix in the likelihood function.
The inversion code is the DREAMzs (ter Braak and Vrugt, 2008; Vrugt, 2009; Laloy and Vrugt, 2012)
MCMC sampler and the considered forward  problem involves 2D GPR linear tomography or Eikonal simulations.

@ original author: Eric Laloy <elaloy@sckcen.be>
@ modifications by Shiran Levy <shiran.levy@unil.ch> June 2021

Please drop me an email if you have any question and/or if you find a bug in this
program. 

===
Copyright (C) 2018  Eric Laloy (original code available here: https://github.com/elaloy/SGANinv)
Copyright (C) 2021 Shiran Levy (modifications)

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

Levy et al. under review. Using deep generative neural networks to account for model
    errors in Markov chain Monte Carlo inversion [submitted to Geophysical Journal 
    International on June 2021].                                    
                                                                                                                                                                                                       
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

CaseStudy = 2   # 1-Eikonal, 2-SGAN-ME, 3-covariance matrix, 4-Ignoring model error, 
                # 5-pixel-to-pixel model error, 6-pixel-to-pixel model
#---------------------------------------------------------------------------------------------------------------------------------
# delayed model error scenario. model error is switched from 0 to 1 at number of iteration in "stage"
delayed_err = 0
stage = 5000
#---------------------------------------------------------------------------------------------------------------------------------
# If to gradually introduce model error contribution, 0-no, 1-yes
weighted_err = 0
#---------------------------------------------------------------------------------------------------------------------------------
#invert a constant to scale model errors (as applied in the SGAN-ME paper)
err_scale=1 
#---------------------------------------------------------------------------------------------------------------------------------
# Enable tempering during burn-in. When tempering is 1 and T>1: Gradual decrease in 
# temperature (power) applied on acceptance ratio of likelihoods
# if tempering is 1 and T=1: only adaptive jr_scale

tempering = 1       # to tune only jr_scale during burn-in, set to 1 with T=1 other wise T>0
minJRS = 0.15       # minimal value for jr_scale
T = 1               # Temperature for annealing
burnin = 20000      # burn-in period (stop tempering)
#---------------------------------------------------------------------------------------------------------------------------------

if  CaseStudy==1: # run MCMC inversion with Eikonal solver
    seq=8              # number of chains
    ndraw=seq*100000   # number of steps per chain
    thin=50            # how often to save the output: Sequences
    steps=10           # how often to calculate the diagnostics
    jr_scale=5         # bigger jumps --> bigger value
    Prior='LHS'        
    
if  CaseStudy==2: # Linear GPR tomography problem: accounting for model errors using SGAN generator
    seq=8
    ndraw=seq*100000
    thin=50            
    steps=10           
    jr_scale=5         
    Prior='LHS'        
    
if CaseStudy==3: # Linear GPR tomography problem: accounting for model errors using covariance matrix (Hansen et al. 2014)
    seq=8
    ndraw=seq*100000
    thin=50
    steps=10
    jr_scale=5
    Prior='LHS'

if CaseStudy==4: # Linear GPR tomography problem: igonoring model errors
    seq=8
    ndraw=seq*100000
    thin=50
    steps=10
    jr_scale=5
    Prior='LHS'
    
if CaseStudy==5: # pixel-to-pixel model error
    seq=8
    ndraw=seq*100000
    thin=10                  
    steps=10                
    jr_scale=0.15
    Prior='LHS'

if CaseStudy==6: # pixel-to-pixel model
    seq=8
    ndraw=seq*1000
    thin=10                  
    steps=10                
    jr_scale=0.15
    Prior='LHS'
    #add choice of log liklihood for each case
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DoParallel=True # forward model is so quick here that parallel is not necessary
parallel_jobs=None

scen = [delayed_err,weighted_err,err_scale]   
if np.sum(scen)>1:
    print('#### Error: you have more than one case for dealing with model errors on, make sure that only one is set to 1 ####')
    sys.exit()

if torch.cuda.is_available():
    device = True
else:
    device = False

#% Run the DREAMzs algorithm
if __name__ == '__main__':
    
    start_time = time.time()

    q=mcmc.Sampler(main_dir=work_dir,CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,Prior=Prior,parallel_jobs=seq,steps=steps,
                   parallelUpdate =0.8 ,pCR=True,thin=thin,nCR=3,DEpairs=1,pJumpRate_one=0.2,BoundHandling='Fold',
                   lik_sigma_est=False,DoParallel=DoParallel,jr_scale=jr_scale,rng_seed=rng_seed,device=device,
                   saveout=True,save_tmp_out=True,tempering=tempering,
                   T=T, burnin=burnin,weighted_err=weighted_err,delayed_err=delayed_err,stage=stage,err_scale=err_scale, minJRS=minJRS)
    
    print("Iterating")
    
    tmpFilePath = None
    #tmpFilePath = './out_tmp.pkl' # None or: work_dir+'\out_tmp.pkl' for a restart
    
    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar, Extra, DNN = q.sample(RestartFilePath=tmpFilePath)
    
    OutDiag, MCMCPar, MCMCVar, Extra, DNN = dict(OutDiag), dict(MCMCPar), dict(MCMCVar), dict(Extra), dict(DNN)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time)) 
