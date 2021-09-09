# -*- coding: utf-8 -*-
"""
Created on Sat May 19 10:04:09 2018

@author: Eric Laloy <elaloy elaloy@sckcen.be>
"""

import numpy as np
from scipy.signal import medfilt

def generate(generator,
             z,
             filtering=False,
             threshold=False,
             tricat=False):

    # Making the prediction
    model = generator(z)# model is a model realization
    
    model=model.detach().cpu().numpy()#if I work on a GPU, move the model realization to the CPU-memory to use numpy or scipy. On GPU you can only use pure pytorch. 

#    model = (model + 1) * 0.5  # Convert from [-1,1] to [0,1]

    # Postprocess if requested
    if filtering:
        for ii in range(model.shape[0]):
            model[ii, :] = medfilt(model[ii, 0,:,:], kernel_size=(3, 3))

    if threshold and not (tricat):
        #        for ii in xrange(model.shape[0]):
        #            threshold=filters.threshold_otsu(model[ii,:])
        #            model[ii,:][model[ii,:]<threshold]=0
        #            model[ii,:][model[ii,:]>=threshold]=1
        threshold = 0.5
        model[model < threshold] = 0
        model[model >= threshold] = 1

    if threshold and tricat:
        model[model < 0.334] = 0
        model[model >= 0.667] = 2
        model[np.where((model > 0) & (model < 2))] = 1
        model = model / 2.0

    return model

if __name__ == "__main__":
    #main()
    generate()
