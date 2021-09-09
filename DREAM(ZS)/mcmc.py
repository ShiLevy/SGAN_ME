# -*- coding: utf-8 -*-
"""
@ original author: Eric Laloy <elaloy@sckcen.be>
@ modifications by Shiran Levy <shiran.levy@unil.ch> June 2021
"""
from __future__ import print_function

import numpy as np
import numpy.matlib as matlib
try:
    import cPickle as pickle
except:
    import pickle

import time

from scipy.stats import triang

from mcmc_func import* # This imports both all Dream_zs and inverse problem-related functions

import sys

from attrdict import AttrDict

MCMCPar=AttrDict()

MCMCVar=AttrDict()

Measurement=AttrDict()

OutDiag=AttrDict()

Extra=AttrDict()

class Sampler:
    
    def __init__(self, main_dir=None,CaseStudy=0,seq = 3,ndraw=10000,thin = 1,  nCR = 3, 
                 DEpairs = 3, parallelUpdate = 0.9, pCR=True,k=10,pJumpRate_one=0.2,
                 steps=100,savemodout=False,saveout=True,save_tmp_out=True,Prior='LHS',
                 DoParallel=True,eps=5e-2,BoundHandling='Reflect',
                 lik_sigma_est=False,parallel_jobs=4,jr_scale=1.0,rng_seed=123,
                 device='cpu',tempering = 0, T = 10, burnin = 20000,
                 weighted_err=0,delayed_err=0,stage=5000,err_scale=0,minJRS=0.1):
        
        self.CaseStudy=CaseStudy
        MCMCPar.seq = seq
        MCMCPar.ndraw=ndraw
        MCMCPar.thin=thin
        MCMCPar.nCR=nCR
        MCMCPar.DEpairs=DEpairs
        MCMCPar.parallelUpdate=parallelUpdate
        MCMCPar.Do_pCR=pCR
        MCMCPar.k=k                                     #Controls the rate of appending to Z and X
        MCMCPar.pJumpRate_one=pJumpRate_one
        MCMCPar.steps=steps
        MCMCPar.savemodout=savemodout
        MCMCPar.saveout=saveout  
        MCMCPar.save_tmp_out=save_tmp_out  
        MCMCPar.Prior=Prior
        MCMCPar.DoParallel=DoParallel
        MCMCPar.eps = eps
        MCMCPar.BoundHandling = BoundHandling
        MCMCPar.lik_sigma_est=lik_sigma_est
        MCMCPar.tempering = tempering
        MCMCPar.jr_scale=jr_scale
        MCMCPar.burnin = burnin
        MCMCPar.T_init = T
        MCMCPar.T = MCMCPar.T_init
        Extra.n_jobs=parallel_jobs
        Extra.main_dir=main_dir
        np.random.seed(rng_seed)
        MCMCPar.rng_seed=rng_seed
        Extra.casestudy = CaseStudy
        MCMCPar.weighted_err=weighted_err
        MCMCPar.delayed_err=delayed_err
        MCMCPar.stage=stage
        MCMCPar.err_scale=err_scale
        MCMCPar.minJRS=minJRS

#%% Defining inversion and simulation parameters
        Model_num = 1
        if self.CaseStudy<5:   

            if CaseStudy == 1:
                ModelName='Eikonal'
            else:
                ModelName='linear_gpr_tomo'

            MCMCPar.lik=2
            MCMCPar.limit_angle = True
            MCMCPar.angle = 50      # max aperture angle
            MCMCPar.savemodout=False
            Extra.err_correction=True
               
            nx=40 # Here x is the horizontal axis (number of columns) and not the number of rows
            ny=61 # Here y is the vertical axis (number of rows) and not the number of columns
            
            units = 0.1  # 0.1 dm
            spacing = 0.1/units
            x = np.arange(0,nx+1,1)*spacing
            y = np.arange(0,ny+1,1)*spacing
            SnR_spacing = 2
            start = 2
            sourcex = 0*spacing
            sourcez = np.arange(start,ny,SnR_spacing)*spacing      #sources positions in meters
            receiverx = 40*spacing
            receiverz = np.arange(start,ny,SnR_spacing)*spacing       #receivers positions in meters
            Extra.nsource = len(sourcez); Extra.nreceiver = len(receiverz)
            ndata=Extra.nsource*Extra.nreceiver
            data=np.zeros((ndata,4))

            if ModelName=='linear_gpr_tomo':
                from tomokernel_straight import tomokernel_straight_2D
                # Calculate acquisition geometry (multiple-offset gather)
                for jj in range(0,Extra.nsource):
                    for ii in range(0,Extra.nreceiver):
                        data[ ( jj ) * Extra.nreceiver + ii , :] = np.array([sourcex, sourcez[jj]+spacing/2, receiverx, receiverz[ii]+spacing/2])
                # Calculate forward modeling kernel (from Matlab code by Dr. James Irving, UNIL)
                G = tomokernel_straight_2D(data,x,y) # Distance of ray-segment in each cell for each ray
                Extra.G=np.array(G.todense())
            else:
                Extra.sx = np.int(sourcex)        #sources positions in model domain coordinates
                Extra.sz = np.round(sourcez).astype(int)       # divided by the spacing to get the domain coordinate 
                Extra.rx = np.int(receiverx/spacing)      #receivers positions in model domain coordinates
                Extra.rz = np.round(receiverz/spacing).astype(int)      # divided by the spacing to get the domain coordinate
                Extra.spacing = spacing
            
            ''' Limit the angle of the shots'''
            if MCMCPar.limit_angle == True:
                count = 0
                i = 0
                zmin = sourcez[0];  zmax = sourcez[-1]; 
                Extra.index = []
                # Calculate acquisition geometry (multiple-offset gather)
                for jj in range(0,Extra.nsource):
                    for ii in range(0,Extra.nreceiver):
                        aOVERb = abs((receiverz[ii]-sourcez[jj]))/(receiverx-sourcex)
                        count+=1
                        if np.arctan(aOVERb) > (MCMCPar.angle*np.pi/180):            
                            # data = data[:-1,:]
                            Extra.index = np.append(Extra.index,i) 
                        i+=1
                Extra.index = np.asarray(Extra.index).astype(int) 
                            
            # DNN model
            DNN=AttrDict()
            DNN.nx=nx
            DNN.ny=ny
            DNN.zx=5
            DNN.zy=5
            DNN.nc=1
            DNN.nz=1
            DNN.depth=5
            DNN.threshold=False
            DNN.filtering=False
            DNN.cuda=device
            DNN.err_inv = False
            
            DNN.gpath_model=Extra.main_dir+'/netG_model.pth'
            
            from generator import Generator as Generator
            
            DNN.npx=(DNN.zx-1)*2**DNN.depth + 1
            DNN.npy=(DNN.zy-1)*2**DNN.depth + 1
            DNN.netG_model = Generator(cuda=DNN.cuda, gpath=DNN.gpath_model, nz = DNN.nz)
            for param in DNN.netG_model.parameters():
                param.requires_grad = False
            DNN.netG_model.eval()
            if DNN.cuda:
                DNN.netG_model.cuda()
            self.DNN=DNN
            
            MCMCPar.lb=np.ones((1,DNN.zx*DNN.zy*DNN.nz))*-1   # number of parameters to estimate 
            MCMCPar.ub=np.ones((1,DNN.zx*DNN.zy*DNN.nz))      # 9 for the model 
            MCMCPar.n=MCMCPar.ub.shape[1]
            
            if Extra.err_correction:
                Extra.model_correction = np.load('./model_mean_correction.npy')
            
            if self.CaseStudy==2: #accounting for model error with SGAN_ME
                
                DNN.err_inv = True

                DNN.gpath_error=Extra.main_dir+'/netG_moderr.pth'
                
                DNN.netG_error = Generator(cuda=DNN.cuda, gpath=DNN.gpath_error, nz = DNN.nz)
                for param in DNN.netG_error.parameters():
                    param.requires_grad = False
                DNN.netG_error.eval()
                if DNN.cuda:
                    DNN.netG_error.cuda()
                self.DNN=DNN
                               
                MCMCPar.lb=np.append(MCMCPar.lb,np.ones(DNN.zx*DNN.zy*DNN.nz)*-1).reshape(1,-1)   # number of parameters to estimate 
                MCMCPar.ub=np.append(MCMCPar.ub,np.ones(DNN.zx*DNN.zy*DNN.nz)).reshape(1,-1)    # adding model error parameters
                if err_scale:
                    MCMCPar.lb=np.append(MCMCPar.lb,0).reshape(1,-1)   # number of parameters to estimate 
                    MCMCPar.ub=np.append(MCMCPar.ub,2).reshape(1,-1)    # adding model error scale parameter
                MCMCPar.n=MCMCPar.ub.shape[1]
                if Extra.err_correction:
                    Extra.correction = np.load('./merror_mean_correction.npy')

            #Load measurements
            file_name='./Model'+str(Model_num)+'/True_data_sig0_5.npy'
            Measurement.MeasData=np.load(file_name)
            Measurement.Sigma=0.5 # Measurement error standard deviation or mean of error dist. if lik_sigma_est=True
            if MCMCPar.limit_angle:
                Measurement.MeasData = np.delete(Measurement.MeasData,Extra.index, axis=0)
            Measurement.True_model = np.load('./Model'+str(Model_num)+'/True_model.npy') #only for the purpose of plotting results, not used in the inversion
            Measurement.N=len(Measurement.MeasData)
            
            if self.CaseStudy==3: #accounting for model error covariance matrix (Hansen et al. 2014)
                MCMCPar.lik=5
                MCMCPar.n_err = 800
                folder = './model_errors_database/'   #model errors folder
                Extra.d_D, Extra.C_T = mean_n_cov(folder, MCMCPar, Extra)       # model mean and covariance
                if not MCMCPar.lik_sigma_est:
                    Extra.C_d = (Measurement.Sigma**2)*np.identity(Measurement.N)   # measurement *variance*
                    Extra.C_D = Extra.C_T + Extra.C_d                                # combined covariance, eq. 9 in Hensen et al., 2014 
                    signDet, Extra.log_det_Cd = np.linalg.slogdet(Extra.C_D)        # log of det of cov. to avoid numerical problems
                    del signDet                                                     # sign of loag is not needed since we only need log(det|C_D|)
                    Extra.C_D_inv = np.linalg.inv(Extra.C_D)                        # inverse of C_D to save computation later
            
            if MCMCPar.lik_sigma_est:
                
                MCMCPar.lb=np.append(MCMCPar.lb,0.5).reshape(1,-1)   # number of parameters to estimate 
                MCMCPar.ub=np.append(MCMCPar.ub,1.0).reshape(1,-1)    # adding model error parameters
                MCMCPar.n=MCMCPar.ub.shape[1]    
            
        elif self.CaseStudy==5:     #pixel-to-pixel inversion of the model error only
        
            ModelName='model_error'
            MCMCPar.lik=2
            MCMCPar.savemodout=False
            MCMCPar.lik_sigma_est=False     # to estimate the sigma as well or not
            Extra.err_correction=True

            # DNN model
            DNN=AttrDict()
            DNN.zx=5
            DNN.zy=5
            DNN.nc=1
            DNN.nz=1
            DNN.depth=5
            DNN.threshold=False
            DNN.filtering=False
            DNN.cuda=device
            DNN.err_inv = False
            
            from generator import Generator as Generator
            
            DNN.npx=(DNN.zx-1)*2**DNN.depth + 1
            DNN.npy=(DNN.zy-1)*2**DNN.depth + 1
            
            DNN.err_inv = False

            DNN.gpath_error=Extra.main_dir+'/netG_moderr.pth'
            DNN.netG_error = Generator(cuda=DNN.cuda, gpath=DNN.gpath_error, nz = DNN.nz)
            for param in DNN.netG_error.parameters():
                param.requires_grad = False
            DNN.netG_error.eval()
            if DNN.cuda:
                DNN.netG_error.cuda()
            self.DNN=DNN
            
            MCMCPar.lb=np.ones((1,DNN.zx*DNN.zy*DNN.nz))*-1   # number of parameters to estimate 
            MCMCPar.ub=np.ones((1,DNN.zx*DNN.zy*DNN.nz))      # 9 for the model 
            if err_scale:
                MCMCPar.lb=np.append(MCMCPar.lb,0).reshape(1,-1)   # number of parameters to estimate 
                MCMCPar.ub=np.append(MCMCPar.ub,1).reshape(1,-1)    # adding model error parameters
            MCMCPar.n=MCMCPar.ub.shape[1]
            
            #Load measurements
            file_name='./Model'+str(Model_num)+'/True_error.npy'
            Measurement.MeasData=np.load(file_name)
            Measurement.Sigma=0.01                             # Measurement error standard deviation
            Extra.nsource = len(Measurement.MeasData[0,:]); Extra.nreceiver = len(Measurement.MeasData[:,0])
            Measurement.MeasData = Measurement.MeasData.flatten(order = 'F')
            Measurement.N=len(Measurement.MeasData.flatten())
            if Extra.err_correction:
                Extra.correction = np.load('./merror_mean_correction.npy')
                
        elif self.CaseStudy==6:     #pixel-to-pixel inversion of the model only
        
            ModelName='model'
            MCMCPar.lik=2
            MCMCPar.savemodout=False
            MCMCPar.lik_sigma_est=False     # to estimate the sigma as well or not
            Extra.err_correction = True
            
            nx=40 # Here x is the horizontal axis (number of columns) and not the number of rows
            ny=61 # Here y is the vertical axis (number of rows) and not the number of columns
            # DNN model
            DNN=AttrDict()
            DNN.nx=nx
            DNN.ny=ny
            DNN.zx=5
            DNN.zy=5
            DNN.nc=1
            DNN.nz=1
            DNN.depth=5
            DNN.threshold=False
            DNN.filtering=False
            DNN.cuda=device
            DNN.err_inv = False
            ''' add the file path of the model error as well'''
            
            from generator import Generator as Generator
            
            DNN.gpath_model=Extra.main_dir+'/netG_model.pth'
            DNN.netG_model = Generator(cuda=DNN.cuda, gpath=DNN.gpath_model, nz = DNN.nz)
            for param in DNN.netG_model.parameters():
                param.requires_grad = False
            DNN.netG_model.eval()
            if DNN.cuda:
                DNN.netG_model.cuda()
            self.DNN=DNN

            MCMCPar.lb=np.ones((1,DNN.zx*DNN.zy*DNN.nz))*-1   # number of parameters to estimate 
            MCMCPar.ub=np.ones((1,DNN.zx*DNN.zy*DNN.nz))      # 9 for the model 
            MCMCPar.n=MCMCPar.ub.shape[1]

            if Extra.err_correction:
                Extra.model_correction = np.load('./model_mean_correction.npy')
            
            #Load measurements
            file_name='./Model'+str(Model_num)+'/True_model.npy'
            Measurement.MeasData=np.load(file_name)
            Measurement.Sigma=0.01                             # Measurement error standard deviation
            Measurement.N=len(Measurement.MeasData.flatten())

        #initial size of the archive matrix Z
        MCMCPar.m0=10*MCMCPar.n
        
        self.MCMCPar=MCMCPar
        self.Measurement=Measurement
        self.Extra=Extra
        self.ModelName=ModelName

#%% Initialization

    def _init_sampling(self):
        
        Iter=self.MCMCPar.seq
        iteration=2
        iloc=0
        T=0
        
        if self.MCMCPar.Prior=='StandardNormal':
            Zinit=np.random.randn(self.MCMCPar.m0+self.MCMCPar.seq,self.MCMCPar.n)
            if self.MCMCPar.lik_sigma_est==True: # Use log-uniform prior for sigma
                Zinit[:,0]=lhs(self.MCMCPar.lb[0][0].reshape((1,1)),self.MCMCPar.ub[0][0].reshape((1,1)),self.MCMCPar.m0+self.MCMCPar.seq, Measurement.Sigma).reshape((self.MCMCPar.m0+self.MCMCPar.seq))
                
        elif self.MCMCPar.Prior=='COV': # Generate initial population from multivariate normal distribution but the model returns posterior density directly
            Zinit=np.random.randn(self.MCMCPar.m0+self.MCMCPar.seq,self.MCMCPar.n)
        
        else: # Uniform prior, LHS sampling
            Zinit=lhs(self.MCMCPar.lb,self.MCMCPar.ub,self.MCMCPar.m0+self.MCMCPar.seq,Measurement.Sigma, MCMCPar.lik_sigma_est)
            
        self.MCMCPar.CR=np.cumsum((1.0/self.MCMCPar.nCR)*np.ones((1,self.MCMCPar.nCR)))
        Nelem=np.floor(self.MCMCPar.ndraw/self.MCMCPar.seq)++self.MCMCPar.seq*2
        OutDiag.CR=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.nCR+1))
        OutDiag.AR=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,2))
        OutDiag.AR[0,:] = np.array([self.MCMCPar.seq,-1])
        OutDiag.R_stat = np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.n+1))
        pCR = (1.0/self.MCMCPar.nCR) * np.ones((1,self.MCMCPar.nCR))
        
        if MCMCPar.tempering:
            OutDiag.T=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,2))
            OutDiag.T[0,:] = np.array([self.MCMCPar.seq,self.MCMCPar.T_init])
            OutDiag.jr_scale=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,2))
            OutDiag.jr_scale[0,:] = np.array([self.MCMCPar.seq,self.MCMCPar.jr_scale])
                    
        # Calculate the actual CR values based on pCR
        CR,lCR = GenCR(self.MCMCPar,pCR)  
        
        if self.MCMCPar.savemodout:
            self.fx = np.zeros((self.Measurement.N,np.int(np.floor(self.MCMCPar.ndraw/self.MCMCPar.thin))))
            MCMCVar.m_func = self.MCMCPar.seq     
        
        self.Sequences = np.zeros((np.int(np.floor(Nelem/self.MCMCPar.thin)),self.MCMCPar.n+2,self.MCMCPar.seq))
           
        self.MCMCPar.Table_JumpRate=np.zeros((self.MCMCPar.n,self.MCMCPar.DEpairs))
        for zz in range(0,self.MCMCPar.DEpairs):
            self.MCMCPar.Table_JumpRate[:,zz] = 2.38/np.sqrt(2 * (zz+1) * np.linspace(1,self.MCMCPar.n,self.MCMCPar.n).T)
        
        # Change steps to make sure to get nice iteration numbers in first loop
        self.MCMCPar.steps = self.MCMCPar.steps - 1
        
        self.Z = np.zeros((np.floor(self.MCMCPar.m0 + self.MCMCPar.seq * (self.MCMCPar.ndraw - self.MCMCPar.m0) / (self.MCMCPar.seq * self.MCMCPar.k)).astype('int64')+self.MCMCPar.seq*100,self.MCMCPar.n+2))
        self.Z[:self.MCMCPar.m0,:self.MCMCPar.n] = Zinit[:self.MCMCPar.m0,:self.MCMCPar.n]

        X = Zinit[self.MCMCPar.m0:(self.MCMCPar.m0+self.MCMCPar.seq),:self.MCMCPar.n]
        # print(np.shape(X))
        ###X[0,:]=Extra.z_true.cpu().numpy().flatten()
        self.Extra.RMSE = np.empty([MCMCPar.seq,0])
        self.Extra.of_acc = np.empty([MCMCPar.seq,0])
        
        del Zinit
        MCMCVar.Iter=Iter
        # Run forward model, if any this is done in parallel
        if  self.CaseStudy > 0:
            if self.MCMCPar.lik_sigma_est==True: # The inferred sigma must always occupy the last position in the parameter vector
                fx0, Extra = RunFoward(X[:,:-1],self.MCMCPar,MCMCVar.Iter,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN) 
            else:
                fx0, Extra = RunFoward(X,self.MCMCPar,MCMCVar.Iter,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)    
        else:
            fx0, Extra = RunFoward(X,self.MCMCPar,MCMCVar.Iter,self.Measurement,self.ModelName,self.Extra)
        
        # Compute likelihood from simulated data
        of,log_p = CompLikelihood(X,fx0,self.MCMCPar,self.Measurement,self.Extra)

        X = np.concatenate((X,of,log_p),axis=1)
        Xfx = fx0
        
        if self.MCMCPar.savemodout==True:
            self.fx=fx0
        else:
            self.fx=None

        self.Sequences[0,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))

        # Store N_CR
        OutDiag.CR[0,:MCMCPar.nCR+1] = np.concatenate((np.array([Iter]).reshape((1,1)),pCR),axis=1)
        delta_tot = np.zeros((1,self.MCMCPar.nCR))

        # Compute the R-statistic of Gelman and Rubin
        OutDiag.R_stat[0,:self.MCMCPar.n+1] = np.concatenate((np.array([Iter]).reshape((1,1)),GelmanRubin(self.Sequences[:1,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)),axis=1)
      
        self.OutDiag=OutDiag
        
        # Also return the necessary variable parameters
        MCMCVar.m=self.MCMCPar.m0
        MCMCVar.iteration=iteration
        MCMCVar.iloc=iloc; MCMCVar.T=T; MCMCVar.X=X
        MCMCVar.Xfx=Xfx; MCMCVar.CR=CR; MCMCVar.pCR=pCR
        MCMCVar.lCR=lCR; MCMCVar.delta_tot=delta_tot
        self.MCMCVar=MCMCVar
        
        if self.MCMCPar.save_tmp_out==True:
            with open('./out_tmp.pkl','wb') as f:
            #with open('./out_tmp.pkl','wb') as f:

                 pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                 'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                 'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                 'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

#%% start sampling --> inversion

    def sample(self,RestartFilePath=None):
        
        if not(RestartFilePath is None):
            print('This is a restart')
            with open(RestartFilePath, 'rb') as fin:
                tmp_obj = pickle.load(fin)
            self.Sequences=tmp_obj['Sequences']
            self.Z=tmp_obj['Z']
            self.OutDiag=tmp_obj['OutDiag']
            self.fx=tmp_obj['fx']
            self.MCMCPar=tmp_obj['MCMCPar']
            self.MCMCVar=tmp_obj['MCMCVar']
            self.Measurement=tmp_obj['Measurement']
            self.ModelName=tmp_obj['ModelName']
            self.Extra=tmp_obj['Extra']
            del tmp_obj
            
            self.ndim=self.MCMCPar.n
#                
            self.MCMCPar.ndraw = 2 * self.MCMCPar.ndraw
            
            # Reset rng
            np.random.seed(np.floor(time.time()).astype('int'))
            
            # Extend Sequences, Z, OutDiag.AR,OutDiag.Rstat and OutDiag.CR
            self.Sequences=np.concatenate((self.Sequences,np.zeros((self.Sequences.shape))),axis=0)
            self.Z=np.concatenate((self.Z,np.zeros((self.Z.shape))),axis=0)
            self.OutDiag.AR=np.concatenate((self.OutDiag.AR,np.zeros((self.OutDiag.AR.shape))),axis=0)
            self.OutDiag.R_stat=np.concatenate((self.OutDiag.R_stat,np.zeros((self.OutDiag.R_stat.shape))),axis=0)
            self.OutDiag.CR=np.concatenate((self.OutDiag.CR,np.zeros((self.OutDiag.CR.shape))),axis=0)
            if MCMCPar.tempering:
                self.OutDiag.T=np.concatenate((self.OutDiag.T,np.zeros((self.OutDiag.T.shape))),axis=0)
                self.OutDiag.jr_scale=np.concatenate((self.OutDiag.jr_scale,np.zeros((self.OutDiag.jr_scale.shape))),axis=0)
            
        else:
            self._init_sampling()
            
        # Main sampling loop  
        print('Iter =',self.MCMCVar.Iter)
        while self.MCMCVar.Iter < self.MCMCPar.ndraw:
            
            # Check that exactly MCMCPar.ndraw are done (uneven numbers this is impossible, but as close as possible)
            if (self.MCMCPar.steps * self.MCMCPar.seq) > self.MCMCPar.ndraw - self.MCMCVar.Iter:
                # Change MCMCPar.steps in last iteration 
                self.MCMCPar.steps = np.ceil((self.MCMCPar.ndraw - self.MCMCVar.Iter)/np.float(self.MCMCPar.seq)).astype('int64')
                
            # Initialize totaccept
            totaccept = 0

#            start_time = time.time()
            
            # Loop a number of times before calculating convergence diagnostic, etc.
            for gen_number in range(0,self.MCMCPar.steps):
                
                # Update T
                self.MCMCVar.T = self.MCMCVar.T + 1
                
                # Define the current locations and associated log-densities
                xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])
                log_p_xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])

                # Without replacement draw rows from Z for proposal creation
                R=np.random.permutation(self.MCMCVar.m)
                R=R[0:2 * self.MCMCPar.DEpairs * self.MCMCPar.seq]
                Zoff = np.array(self.Z[R,:self.MCMCPar.n])
             
        
                # Determine to do parallel direction or snooker update
                if (np.random.rand(1) > self.MCMCPar.parallelUpdate) and (self.MCMCVar.Iter/self.MCMCPar.seq) <= self.MCMCPar.burnin:
                    Update = 'Snooker_Update'
                else:
                    Update = 'Parallel_Direction_Update'
                    
                # Generate candidate points (proposal) in each chain using either snooker or parallel direction update
                xnew,self.MCMCVar.CR[:,gen_number] ,alfa_s = DreamzsProp(xold,Zoff,self.MCMCVar.CR[:,gen_number],self.MCMCPar,Update)
    
    
                # Get simulated data (done in parallel)
                if  self.CaseStudy > 0:
                    if self.MCMCPar.lik_sigma_est==True: # The inferred sigma must always occupy the last position in the parameter vector
                        fx_new, Extra = RunFoward(xnew[:,:-1],self.MCMCPar,self.MCMCVar.Iter,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
                    else:
                        fx_new, Extra = RunFoward(xnew,self.MCMCPar,self.MCMCVar.Iter,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)    
                else:
                    fx_new, Extra = RunFoward(xnew,self.MCMCPar,self.MCMCVar.Iter,self.Measurement,self.ModelName,self.Extra)
                 
                # Compute the likelihood of each proposal in each chain
                of_xnew,log_p_xnew = CompLikelihood(xnew,fx_new,self.MCMCPar,self.Measurement,self.Extra)
                
                # Calculate the Metropolis ratio
                accept = Metrop(self.MCMCPar,self.MCMCVar.Iter,xnew,log_p_xnew,xold,log_p_xold,alfa_s,Extra)

                # And update X and the model simulation
                idx_X= np.argwhere(accept==1);idx_X=idx_X[:,0]
                
                if not(idx_X.size==0):
                     
                    self.MCMCVar.X[idx_X,:] = np.concatenate((xnew[idx_X,:],of_xnew[idx_X,:],log_p_xnew[idx_X,:]),axis=1)
                    self.MCMCVar.Xfx[idx_X,:] = fx_new[idx_X,:]
                                  
                # Check whether to add the current points to the chains or not?
                if self.MCMCVar.T == self.MCMCPar.thin:
                    # Store the current sample in Sequences
                    self.MCMCVar.iloc = self.MCMCVar.iloc + 1
                    self.Sequences[self.MCMCVar.iloc,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(self.MCMCVar.X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))
                   
                   # Check whether to store the simulation results of the function evaluations
                    if self.MCMCPar.savemodout==True:
                        self.fx=np.append(self.fx,self.MCMCVar.Xfx,axis=0)
                        # Update m_func
                        self.MCMCVar.m_func = self.MCMCVar.m_func + self.MCMCPar.seq
                    else:
                        self.MCMCVar.m_func=None
                    # And set the T to 0
                    self.MCMCVar.T = 0

                # Compute squared jumping distance for each CR value
                if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):
                   
                    # Calculate the standard deviation of each dimension of X
                    r = matlib.repmat(np.std(self.MCMCVar.X[:,:self.MCMCPar.n],axis=0),self.MCMCPar.seq,1)
                    # Compute the Euclidean distance between new X and old X
                    delta_normX = np.sum(np.power((xold[:,:self.MCMCPar.n] - self.MCMCVar.X[:,:self.MCMCPar.n])/r,2),axis=1)
                                        
                    # Use this information to update delta_tot which will be used to update the pCR values
                    self.MCMCVar.delta_tot = CalcDelta(self.MCMCPar.nCR,self.MCMCVar.delta_tot,delta_normX,self.MCMCVar.CR[:,gen_number])

                # Check whether to append X to Z
                if np.mod((gen_number+1),self.MCMCPar.k) == 0:
                   
                    ## Append X to Z
                    self.Z[self.MCMCVar.m + 0 : self.MCMCVar.m + self.MCMCPar.seq,:self.MCMCPar.n+2] = np.array(self.MCMCVar.X[:,:self.MCMCPar.n+2])
                    # Update MCMCPar.m
                    self.MCMCVar.m = self.MCMCVar.m + self.MCMCPar.seq

                # Compute number of accepted moves
                totaccept = totaccept + np.sum(accept)

                # Update total number of MCMC iterations
                self.MCMCVar.Iter = self.MCMCVar.Iter + self.MCMCPar.seq
                
            print('Iter =',self.MCMCVar.Iter)  
            
            # Reduce MCMCPar.steps to get rounded iteration numbers
            if self.MCMCVar.iteration == 2: 
                self.MCMCPar.steps = self.MCMCPar.steps + 1

            # Store acceptance rate
            self.OutDiag.AR[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)]).reshape((1,1))),axis=1)
            
            # If tempering is on (true only during burn-in), apply it:
            if MCMCPar.tempering:
                if (self.MCMCVar.Iter/self.MCMCPar.seq) <= self.MCMCPar.burnin and (self.MCMCVar.Iter/self.MCMCPar.seq)%100==0 and self.MCMCPar.jr_scale>MCMCPar.minJRS:               
                    # check if the acceptance rate is lower than 20%, if it is decrease the jump
                    if self.OutDiag.AR[self.MCMCVar.iteration-1,1] < 20:
                        self.MCMCPar.jr_scale = 0.9 * self.MCMCPar.jr_scale
                        self.OutDiag.jr_scale[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([self.MCMCPar.jr_scale]).reshape((1,1))), axis=1)
                    # check if the acceptance rate is higher than 30%, if it is increase the jump
                    elif self.OutDiag.AR[self.MCMCVar.iteration-1,1] > 30:
                        self.MCMCPar.jr_scale = (1/0.9) * self.MCMCPar.jr_scale
                        self.OutDiag.jr_scale[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([self.MCMCPar.jr_scale]).reshape((1,1))), axis=1)
                    else:
                        self.OutDiag.jr_scale[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([self.MCMCPar.jr_scale]).reshape((1,1))), axis=1)
                    # Decrease the temperature with iterations during burn-in period (only working if temperature is set to be larger than 1)
                    self.OutDiag.T[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array((self.MCMCPar.T_init + (self.MCMCVar.Iter/self.MCMCPar.seq) * ((1-self.MCMCPar.T_init)/self.MCMCPar.burnin))).reshape((1,1))),axis=1)       
                    self.MCMCPar.T = self.OutDiag.T[self.MCMCVar.iteration-1,1]
                else:
                    self.OutDiag.jr_scale[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([self.MCMCPar.jr_scale]).reshape((1,1))), axis=1)
                    self.OutDiag.T[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)),np.array([self.MCMCPar.T]).reshape((1,1))), axis=1)
                    self.MCMCPar.T = self.OutDiag.T[self.MCMCVar.iteration-1,1]
                
            # Store probability of individual crossover values
            self.OutDiag.CR[self.MCMCVar.iteration-1,:self.MCMCPar.nCR+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), self.MCMCVar.pCR),axis=1)
            
            # Is pCR updating required?
            if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):

                # Update pCR values
                self.MCMCVar.pCR = AdaptpCR(self.MCMCPar.seq,self.MCMCVar.delta_tot,self.MCMCVar.lCR,self.MCMCVar.pCR)

            # Generate CR values from current pCR values
            self.MCMCVar.CR,lCRnew = GenCR(MCMCPar,self.MCMCVar.pCR); self.MCMCVar.lCR = self.MCMCVar.lCR + lCRnew

            # Calculate Gelman and Rubin Convergence Diagnostic
            start_idx = np.maximum(1,np.floor(0.5*self.MCMCVar.iloc)).astype('int64')-1; end_idx = self.MCMCVar.iloc
            
            current_R_stat = GelmanRubin(self.Sequences[start_idx:end_idx,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)
            
            self.OutDiag.R_stat[self.MCMCVar.iteration-1,:self.MCMCPar.n+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)),np.array([current_R_stat]).reshape((1,self.MCMCPar.n))),axis=1)

            # Update number of complete generation loops
            self.MCMCVar.iteration = self.MCMCVar.iteration + 1

            if self.MCMCPar.save_tmp_out==True:
                with open('./out_tmp.pkl','wb') as f:
                #with open('./out_tmp.pkl','wb') as f:
                    pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                    'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                    'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                    'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

        # Remove zeros from pre-allocated variavbles if needed
        self.Sequences,self.Z,self.OutDiag,self.fx = Dreamzs_finalize(self.MCMCPar,self.Sequences,self.Z,self.OutDiag,self.fx,self.MCMCVar.iteration,self.MCMCVar.iloc,self.MCMCVar.pCR,self.MCMCVar.m,self.MCMCVar.m_func)
        
        if self.MCMCPar.saveout==True:
            #with open('./dreamzs_out.pkl','wb') as f:
            with open('./dreamzs_out'+'.pkl','wb') as f:
                pickle.dump({'Sequences':self.Sequences,'Z':self.Z,'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,'Measurement':self.Measurement,'Extra':self.Extra},f
                , protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.Sequences, self.Z, self.OutDiag,  self.fx, self.MCMCPar, self.MCMCVar, self.Extra  , self.DNN  
