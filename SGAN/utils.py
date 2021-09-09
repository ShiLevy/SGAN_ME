# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@ original author: Eric Laloy <elaloy@sckcen.be> (original code available here:
https://github.com/elaloy/SGANinv/tree/master/braided_river_pytorch)
    
@ modifications by Shiran Levy <shiran.levy@unil.ch> June 2021
"""

import os
import numpy as np
import re
from PIL import Image
from PIL.Image import FLIP_LEFT_RIGHT
import h5py
from struct import unpack, pack
import matplotlib.pyplot as plt
from tqdm import tqdm
#import copy
from scipy.interpolate import interp2d as int2d
from scipy.interpolate import griddata as intNN
    
def image_to_tensor(img):
    '''
    convert image to Theano/Lasagne 3-tensor format;
    changes channel dimension to be in the first position and rescales from [0,255] to [-1,1]
    '''
#    tensor = np.array(img).transpose( (2,0,1) )
#    tensor = tensor / 128. - 1.
    i_array=np.array(img)
    if len(i_array.shape)==2: # the array is 2d, convert to a 3D array
        i_array=i_array.reshape((i_array.shape[0],i_array.shape[1],1))
    tensor = i_array.transpose( (2,0,1) )
    tensor = tensor / 128. - 1.0
    return tensor


def tensor_to_2Dimage(tensor):
    '''
    convert 3-tensor to image;
    changes channel to be last and rescales from [-1, 1] to [0, 255]
    '''
    img = np.array(tensor).transpose( (1,2,0) )
    img = (img + 1.) * 128.
    return np.uint8(img)
    
#Randomly sample the training image to get a minibatch
def get_texture2D_iter(folder, npx=129, npy=129, batch_size=32,
                     filter=None, n_channel=1):
    HW1   = npy
    HW2   = npx
    imTex = []
    files = os.listdir(folder)
    fmat = files[0].split('.')[-1]
    if fmat == 'png' or fmat == 'jpg':  #for reading images
        for f in files:
            name = folder + f
            try:
                img = Image.open(name)
                imTex += [image_to_tensor(img)]
            except:
                print("Image ", name, " failed to load!")
    else:
        for f in files:     #for reading matrices from file
            name = folder + f
            try:
                img = np.load(name)
                # rescaling the matrix to [-1,1]
                imgBig = 2*((img-np.min(img.flatten()))/(np.max(img.flatten())-np.min(img.flatten())))-1
            except:
                print("Image ", name, " failed to load!")
    while True:
        data=np.zeros((batch_size,n_channel,npy,npx))                   
        for chan in range(n_channel):
            for i in range(batch_size):
                if HW1 < imgBig.shape[0] and HW2 < imgBig.shape[1]:
                    h = np.random.randint(imgBig.shape[0] - HW1)
                    w = np.random.randint(imgBig.shape[1] - HW2)
                    img = imgBig[h:h + HW1, w:w + HW2]
                else:                                               
                    img = imgBig
            
                # try:
                data[i,chan] = img  
                # except:
                #     print("Original image (",origdimx,'x',origdimy,") is larger than npx, npy ","(",npx,'x',npy,")")
                #     break
                        
        yield data
        
def get_texture2D_samples(folder, coo_mat, npx=128, npy=128,batch_size=64, \
                     filter=None, n_channel=1):#modify it
    '''
   still need to document this one but here we get the batches from a given
   set of training images that already have the required npx x npy size
    '''
    imTex = []
    files = os.listdir(folder)
    for f in files:
        name = folder + f
        try:
            with h5py.File(name, 'r') as fid:
                img=np.array(fid['features']) # the 2D training images are already under (channel, W, H) format and [0,1] scale
			# img needs to be rescaled from [0,1] to [-1,1]
            img=((img*1.0)*2-1).astype('float32')
            imTex += [img]

        except:
            print("Image ", name, " failed to load!")
       
    while True:
        data=np.zeros((batch_size,n_channel,npx,npy))  
        ir = 0
        ivec=np.random.choice(np.arange(0,coo_mat.shape[0]), size=batch_size, replace=False)
        for i in range(batch_size):
            xmin=coo_mat[ivec[i],0]
            xmax=coo_mat[ivec[i],1]
            ymin=coo_mat[ivec[i],2]
            ymax=coo_mat[ivec[i],3]
            
            data[i] = imTex[ir][xmin:xmax+1,ymin:ymax+1]

        yield data


def save_tensor2D(tensor, filename):
    '''
    save a 3-tensor (channel, x, y) to image file
    '''
    img = tensor_to_2Dimage(tensor)
    
    if img.shape[2]==1:
        # print('not now either!')
        img=img.reshape((img.shape[0],img.shape[1]))
        img = Image.fromarray(img).convert('L')
    else:
        img = Image.fromarray(img)
    img.save(filename)
    
    
def zx_to_npx(zx,depth,k,s,p):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    
    for ii in range(depth-1):
        output =s*(zx-1)-2*p+(k-1)+1    # all layer up to the last (excl.)
        zx = output
    output = s*(zx-1)-2*4+(k-1)+1       #last layer architecture

    return output    

if __name__=="__main__":
    print("nothing here.")

        
#Randomly sample the database to get a minibatch of errors and interpolate it to a desired size
#scaling is by dividing by a scale factor
def get_err_iter_int(folder_err, scalefac, dimout, batch_size=64, n_channel=1):
    '''
    @param folder_err   iterate of errors from this folder
    @param scalefac     scaling the data between [-1,1]
    @param dimout       size of the error image to be interpolated to (assuming square images)
    @param batch_size   number of batches to yield - if None, it yields forever
    @param n_channel    number of channels of image (1=bw, 3=RGB)
    '''
    # Extract the size of the errors
    files_err = os.listdir(folder_err)
    fmat = files_err[0].split('.')[-1]
    fmat = '.'+fmat
    mod_name = re.split(r'[0-9]',files_err[0])[0]
    f = np.load(folder_err+files_err[0])
    dim1_err = len(f[0])
    dim2_err = len(f[1])
    invec = np.linspace(1,dim1_err,dim1_err)
    outvec = np.linspace(1,dim1_err,dimout)
    while True:
        data_err=np.zeros((batch_size,n_channel,dimout,dimout)) 
        for ib in range(batch_size):
            # LOAD AN ERROR
            # Select a random file
            ir = np.random.randint(0,len(files_err))
            # read the error from the file
            f = np.load(folder_err+mod_name+str(ir)+fmat)
            f = f/scalefac
            # Interpolate error image to desired output size
            intfun = int2d(invec,invec,f,kind='linear')
            errmat_int = intfun(outvec,outvec)
            # Add the error to the batch
            data_err[ib] = errmat_int
        
        yield data_err

#Randomly sample the database to get a minibatch of errors and interpolate it to a desired size
#scalling is through min-max normalization
def get_err_iter_int_scale(folder_err, extrema, dimout, batch_size=64, n_channel=1):
    '''
    @param folder_err   iterate of errors from this folder
    @param scalefac     scaling the data between [-1,1]
    @param dimout       size of the error image to be interpolated to (assuming square images)
    @param batch_size   number of batches to yield - if None, it yields forever
    @param n_channel    number of channels of image (1=bw, 3=RGB)
    '''
    # Extract the size of the errors
    files_err = os.listdir(folder_err)
    fmat = files_err[0].split('.')[-1]
    fmat = '.'+fmat
    mod_name = re.split(r'[0-9]',files_err[0])[0]
    f = np.load(folder_err+files_err[0])
    dim1_err = len(f[0])
    dim2_err = len(f[1])
    invec = np.linspace(1,dim1_err,dim1_err)
    outvec = np.linspace(1,dim1_err,dimout)
    while True:
        data_err=np.zeros((batch_size,n_channel,dimout,dimout))
        for ib in range(batch_size):
            # LOAD AN ERROR
            # Select a random file
            ir = np.random.randint(0,len(files_err))
            # read the error from the file
            f = np.load(folder_err+mod_name+str(ir)+fmat)
            f = 2*((f-extrema[0])/(extrema[1]-extrema[0]))-1
            # Interpolate error image to desired output size
            intfun = int2d(invec,invec,f,kind='linear')
            errmat_int = intfun(outvec,outvec)
            # Add the error to the batch
            data_err[ib] = errmat_int       

        yield data_err
        
#Determine the minimum and maximum of all data-files for range adjustment between -1 and 1
def det_err_min_max(folder_err):
    '''
    @param folder_err   iterate of errors from this folder
    '''
    # Extract the size of the errors
    files_err = os.listdir(folder_err)
    fmat = files_err[0].split('.')[-1]
    fmat = '.'+fmat
    mod_name = re.split(r'[0-9]',files_err[0])[0]
    f = np.load(folder_err+files_err[0])
    dim1_err = len(f[0])
    dim2_err = len(f[1])
    extrema = np.zeros(2)
    for ib in range(0,len(files_err)):
        # LOAD AN ERROR
        # read the error from the file
        f = np.load(folder_err+mod_name+str(ib)+fmat)
        if np.min(f.flatten())<extrema[0]:
            extrema[0]=np.min(f.flatten())
        if np.max(f.flatten())>extrema[1]:
            extrema[1]=np.max(f.flatten())
    
    # Determine if the max or the min of the data is larger
    if abs(extrema[0])>abs(extrema[1]):
        scalefac=abs(extrema[0])
    else:
        scalefac=abs(extrema[1])
    return scalefac, extrema

