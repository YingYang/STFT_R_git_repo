# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:10:34 2014
L2 regression

@author: yingyang
"""

import numpy as np
import scipy.io 
from .L2_tsparse import get_MSE_stft_regresion_tsparse                  
        
# ======================================================        
def get_btstrp_stft_regresion_with_active_set \
                              (M,G_list, G_ind, X, 
                               Z0, active_set_z0, active_t_ind_z0,
                               coef_non_zero_mat,
                               path, fname,
                                B = 100,
                                wsize=16, tstep = 4, Rescale_Flag = True):                             
    """
    Bootstrap, only fit the regression coefficients, 
          leave the trial by trial STFT coeffficents in the residual
          When bootstrap, resample trials, rather than time points
          If B is large, the ram can not fit all bootstraped samples, 
           so save them into mat files
          When resampling, correct the residual according to Stine
          h_ii = x_i^T (X^TX)^{-1} x_i
          r / (1-h_ii)^0.5

    Input:
        M, [n_sensors, n_times, n_trials] array of the sensor data
        G, [n_sensors, n_dipoles] forward gain matrix
        X, [n_trials, p],the design matrix, it must include an all 1 colume
        Z0, [n_true_dipoles,n_coefs*p], only regression coefficients
        active_set_z0, boolean, active_set
        active_t_ind_z0, boolean, active temporal indices
        coef_non_zero_mat, more detailed active set
        path, path to save
        fname, fname of the mat files
        B, # samples
        wsize, number of frequence
        tstep, step in time of the stft
        Rescale_Flag, if True, normalize the residuals before sampling
    Output:
        0
        The mat files are written in designated folders
        The bootstrapped sample data are written in 
            path + fname + "_btstrp_%04i.mat" %i 
            ['M_btstrp']
        The paramters are written in 
            path+fname+"_btstrp_params.mat"
        tmp_dict = dict(G = G, X = X, Z = Z0, 
                    active_set_z = active_set_z0,
                    active_t_ind_z = active_t_ind_z0,
                    B= B, wsize = wsize, tstep = tstep, oned_as = 'row')
    """

    n_sensors, n_times, n_trials = M.shape
    n_dipoles = G_list[0].shape[1]
    p = X.shape[1]
    if len(active_set_z0) != n_dipoles:
       raise ValueError("the number of dipoles does not match!")
       
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq
    #phi = _Phi(wsize, tstep, n_coefs)
    #phiT = _PhiT(tstep, n_freq, n_step, n_times)

    _, residual_M, _, _ = get_MSE_stft_regresion_tsparse(M,G_list, G_ind,X,
                                Z0, active_set_z0, active_t_ind_z0,
                                wsize=wsize, tstep = tstep)               
    # since residual = M - predicted_M
    predicted_M = M - residual_M  
    norm_scale = np.ones(n_trials)
    invXTX = np.linalg.inv(X.T.dot(X))
    if Rescale_Flag:
        for i in range(n_trials):
            norm_scale0 = np.dot(np.dot(invXTX, X[i,:]).T, X[i,:])
            # norm_scale0 must be smaller then 1
            if norm_scale0 > 1:
                raise ValueError("Bootstrap weight greater than 1")
            norm_scale[i] = np.sqrt((1.0-norm_scale0))
    # else, norm_scale = 1.0, if X is a full rank square matrix, then it is always 1
    # i.e. n>p must hold!
    rescaled_residual = residual_M/norm_scale
    # set the seed first
    for i in range(B):
        rd_index = np.random.randint(n_trials, size = n_trials)
        #btstrp_residual_M = (rescaled_residual[:,:,rd_index])*norm_scale
        # changed on Oct 6, I was wrong about the scaling, no need to scale back
        btstrp_residual_M = (rescaled_residual[:,:,rd_index])
        btstrp_M = btstrp_residual_M + predicted_M
        
        tmp_dict = dict(btstrp_M = btstrp_M)
        fname_str = path+fname+"_btstrp_%04i.mat" %i 
        scipy.io.savemat(fname_str,tmp_dict, oned_as = 'row')
    
    # save other prameters
    G_list_array = np.zeros(len(G_list), dtype = np.object)
    for run_id in range(len(G_list)):
        G_list_array[run_id] = G_list[run_id]
    tmp_dict = dict(G_list_array = G_list_array, X = X, Z = Z0, 
                    active_set_z = active_set_z0,
                    active_t_ind_z = active_t_ind_z0,
                    B= B, wsize = wsize, tstep = tstep, oned_as = 'row')
    fname_str = path+fname+"_btstrp_params.mat" 
    scipy.io.savemat(fname_str,tmp_dict,oned_as = 'row')
    return 0
    
        
        
        