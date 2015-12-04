# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

@author: ying
"""
import mne
import numpy as np
#import matplotlib.pyplot as plt
from mne.minimum_norm.inverse import (apply_inverse, _check_method, _check_ori,
         prepare_inverse_operator, _pick_channels_inverse_operator, _check_ch_names,
         _assemble_kernel)
from mne.io.constants import FIFF
from mne.time_frequency import stft, istft
import numpy.linalg as la
# ============================================================================
def _apply_inverse_evoked_list(evoked_list, inverse_operator, lambda2, method="MNE",
                              labels=None, nave=1, pick_ori=None,
                              verbose=None, pick_normal=None):
    """ Utility function for applying the inverse solution to a list of evoked object
        Assume that the info for each evoked object in the list is the same
        Input:
            evoked_list, 
            inverse_operator,
            lambda2, 
            method,
            labels, list of label objects
            nave = 1,
            pick_ori = None,
            verbos = none,
            pick_normal = None
        Output: stc_Data, [n_sources_labels, n_times, n_trials]
    """
    info = evoked_list[0].info
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori, pick_normal)
    _check_ch_names(inverse_operator, info)
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    sel = _pick_channels_inverse_operator(info['ch_names'], inv)
    labels_union = None
    if labels is not None:
        labels_union = labels[0]
        if len(labels) > 1:
            for i in range(1,len(labels)):
                labels_union += labels[i]
    K, noise_norm, vertno = _assemble_kernel(inv, labels_union, method, pick_ori)
    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and pick_ori is None)
    if not is_free_ori and noise_norm is not None:
        # premultiply kernel with noise normalization
        K *= noise_norm
    n_channels = len(sel)
    n_times = len(evoked_list[0].times)
    n_trials = len(evoked_list)
    n_sources = K.shape[0]
    stc_Data = np.zeros([n_sources,n_times, n_trials])
    for i in range(n_trials):
        if is_free_ori:
            # Compute solution and combine current components (non-linear)
            sol = np.dot(K, evoked_list[i].data)  # apply imaging kernel
            if is_free_ori:
                logger.info('combining the current components...')
                sol = combine_xyz(sol)
                if noise_norm is not None:
                    sol *= noise_norm
        else:
            # Linear inverse: do computation here or delayed
            sol = np.dot(K, evoked_list[i].data)
        stc_Data[:,:,i] = sol
    return stc_Data
# ===========================================================================
def mne_stft_regression(evoked_list, inverse_operator, X,
                        labels = None, pick_ori=None, pick_normal=None,
                        snr=1, wsize = 64, tstep = 4, Flag_reg_stats = False,
                        method = "MNE"):
    ''' Get the MNE solution for a given snr(lambda value)
    Input:
        evoked_list, a list of evoked instances
        inverse_operator, the inverse operator for MNE
        X, [n_trials, p]  array
        labels, ROI labels list, if None, use the whole brain
        snr, controls lambda
        wsize, window size of the stft transform
        tstep, time step of the stft transform
        method, "MNE", "dSPM", "sLORETA", 
              Note that dSPM and sLORETA can not be used for prediction,
              and the coefficients are normalized too. 
    Output:
        result_dict = dict(coef = coef, F = F, sel = sel,roi_data = roi_data)
        ['coef']: Regression coefficients, complex arrays [n_dipoles,n_coefs,n_steps,p]
        ['F'],F-statistics, complex arrays
        ['sel'], selction of the source points, columns of G
        ['roi_data'] the source data in the ROI
    '''
    n_trials = len(evoked_list)
    sel = []
    # The following line is wrong
    n_dipoles = inverse_operator['nsource']
    # if label is specified, only do the regression on the labels
    # otherwise, select the data for the whole brain. 
    if labels is not None:
        for i in range(len(labels)):
            _, sel_tmp = mne.source_space.label_src_vertno_sel(labels[i],inverse_operator['src'])
            sel = np.hstack([sel, sel_tmp])
            sel = sel.astype(np.int)
    else:
        sel = np.arange(0,n_dipoles,1)
        sel.astype(np.int)
    # tested, the result is the same as running apply_inverse()
    roi_data = _apply_inverse_evoked_list(evoked_list, inverse_operator, 
                                          lambda2= 1.0/snr**2, method=method,
                              labels=labels, nave=1, pick_ori=pick_ori,
                              verbose=None, pick_normal=None)
    n_dipoles, n_times = roi_data[0].shape
    n_trials = len(evoked_list)
    # stft transform, F means the coefficients
    F_roi_data = list()
    for i in range(n_trials):
        F_roi_data.append(stft(roi_data[:,:,i], wsize= wsize, tstep = tstep))
    # put the stft transform into a matrix
    dim0,dim1,dim2 = F_roi_data[0].shape
    F_roi_data_3d = np.zeros([dim0,dim1,dim2,n_trials],dtype = np.complex)
    for i in range(n_trials):
        F_roi_data_3d[:,:,:,i] = F_roi_data[i]
    del(F_roi_data)
    # regression, return coefficients and F-values 
    p = X.shape[1] 
    coef = np.zeros([dim0,dim1,dim2,p], dtype = np.complex)
    F = np.zeros([dim0,dim1,dim2], dtype = np.complex) if Flag_reg_stats else None
    linreg_op = np.dot(la.inv(X.T.dot(X)),X.T)
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                tmpY = np.real(F_roi_data_3d[i,j,k,:])
                tmp_coef = linreg_op.dot(tmpY)
                # debug
                #tmp_coef2 = np.linalg.lstsq(X,tmpY)[0]
                #print np.linalg.norm(tmp_coef-tmp_coef2)
                coef[i,j,k,:] += tmp_coef
                if Flag_reg_stats:
                    tmpY_hat = np.dot(X,tmp_coef)
                    tmp_res = tmpY_hat-tmpY
                    SSE = np.dot(tmp_res,tmp_res)
                    SST =  np.sum((tmpY-np.mean(tmpY))**2)
                    if SSE== 0:
                        F[i,j,k] += 0
                    else:
                        F[i,j,k] += (SST-SSE)/(p-1)/(SSE/(n_trials-p))
                # imaginary
                tmpY = np.imag(F_roi_data_3d[i,j,k,:])
                tmp_coef = linreg_op.dot(tmpY)
                coef[i,j,k,:] += tmp_coef*1j
                if Flag_reg_stats:
                    tmpY_hat = np.dot(X,tmp_coef)
                    tmp_res = tmpY_hat-tmpY
                    SSE = np.dot(tmp_res,tmp_res)
                    SST =  np.sum((tmpY-np.mean(tmpY))**2)
                    if SSE== 0:
                        F[i,j,k] += 0
                    else:
                        F[i,j,k] += (SST-SSE)/(p-1)/(SSE/(n_trials-p))*1j 
    result_dict = dict(coef = coef, F = F, sel = sel,roi_data_3D = roi_data)
    return result_dict

#===============================================================
def get_MSE_mne_stft_regression(evoked_list, fwd, X, coef, labels,
                                wsize = 64, tstep = 4):
    '''
    Use the mne regression coefficients to get predicted sensor data, 
    then abtain the sum of squared error
    
    Input:
        evoked_list, a list of evoked objects
        fwd, the forward solution
        X, the design matrix,
        coef, the regression coefficients, [n_dipoles,n_coefs,n_steps,p]
        wsize, STFT window size 
        tstep, STFT time step
    Output:
        MSE, the sum of squared error across trials
    '''
    sel = []
    n_dipoles = fwd['nsource']
    if labels is not None:
        for i in range(len(labels)):
            _, sel_tmp = mne.source_space.label_src_vertno_sel(labels[i],fwd['src'])
            sel = np.hstack([sel, sel_tmp])
            sel = sel.astype(np.int)
    else:
        sel = np.arange(0,n_dipoles,1)
        sel.astype(np.int)
    
    
    # prepair the forward solution
    evoked_ch_names = evoked_list[0].info['ch_names']
    fwd_ch_names = fwd['info']['ch_names']
    channel_sel = [i for i in range(len(fwd_ch_names))  \
              if fwd_ch_names[i] in evoked_ch_names]
    G = fwd['sol']['data'][channel_sel,:]
    G = G[:,sel]
    ntimes = len(evoked_list[0].times)
    
    n_trials,p = X.shape
    if n_trials != len(evoked_list):
        raise ValueError("the numbers of trials do not match")   
    SSE = 0.0
    for r in range(n_trials):
        # STFT coefficients of current trial
        #predicted_stft_coef = np.zeros(coef.shape[0:3], dtype = np.complex)
        #for j in range(p):   
        #    predicted_stft_coef += coef[:,:,:,j]*X[r,j]
        predicted_stft_coef = np.sum(coef*X[r,:],axis = 3)
        # istft
        predicted_sensor = G.dot(np.real(istft(predicted_stft_coef, tstep = tstep, Tx = ntimes)))
        SSE += np.sum((evoked_list[r].data - predicted_sensor)**2)
    MSE = SSE/(n_trials)
    return MSE       


# ==============================================================
def select_lambda_tuning_mne_stft_regression_cv(evoked_list, inverse_operator,
                                                fwd, X,  cv_partition_ind,
                                                snr_tuning_seq, 
                                                labels = None, 
                                                wsize=64, tstep = 4):
    '''
    Use cross-validation to select the best lambda (tuning snr values)
    All source points across the whole brain must be used,
    This may require a large membory
    Input:
        evoked_list, n_trials of evoked objects
        inverse_operator, the inverse_operator,
        fwd, the forward solution
        X, [n_trials,p] the design matrix
        cv_partition_ind, [n_trials,] parition index for cross validcation
        snr_tuning_seq, a sequence of "snr" parameter
        wsize, STFT window size
        tstep, STFT time step
    Output:
        best_snr_tuning,  the best snr paramter
        cv_MSE, the cross validated SSE for each snr parameters
    '''
   
    n_fold = len(np.unique(cv_partition_ind))
    # number of tuning paramters
    n_par_tuning = len(snr_tuning_seq)
    cv_MSE = np.ones([len(snr_tuning_seq),n_fold], dtype = np.float)*np.Inf
   
    for j in range(n_fold):
         # partition
        test_trials = np.nonzero(cv_partition_ind == j)[0]
        train_trials = np.nonzero(cv_partition_ind != j)[0]           
        evoked_list_train = [evoked_list[r] for r in range(len(evoked_list)) \
                              if r in train_trials]
        Xtrain = X[train_trials,:]
        evoked_list_test = [evoked_list[r] for r in range(len(evoked_list)) \
                              if r in test_trials]
        Xtest = X[test_trials,:] 
        for i in range(n_par_tuning):
            tmp_snr = snr_tuning_seq[i]
            tmp_result = mne_stft_regression(evoked_list_train, inverse_operator, 
                                     Xtrain,  labels = labels,
                                     snr=tmp_snr, wsize = wsize, tstep = tstep)
            coef = tmp_result['coef']
            # Now do the prediction
            tmp_MSE = get_MSE_mne_stft_regression(evoked_list_test, fwd, Xtest,
                                                  coef,  labels = labels,
                                                  wsize = wsize, tstep = tstep)
            cv_MSE[i,j] = tmp_MSE 
    cv_MSE = cv_MSE.mean(axis = 1) 
    best_ind = np.argmin(cv_MSE)
    snr_tuning_star = snr_tuning_seq[best_ind]
    return snr_tuning_star, cv_MSE
           
                                    
    