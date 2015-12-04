# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

@author: ying
"""
import mne
import numpy as np
import numpy.linalg as la
from mne_stft_regression import _apply_inverse_evoked_list

# ===========================================================================
def mne_regression(evoked_list, inverse_operator, X,
                        labels = None, pick_ori=None, pick_normal=None,
                        snr=1, Flag_reg_stats = False,
                        method = "MNE"):
    ''' Get the MNE solution for a given snr(lambda value)
        regress the time points instead of STFT coefficients
    Input:
        evoked_list, a list of evoked instances
        inverse_operator, the inverse operator for MNE
        X, [n_trials, p]  array
        labels, ROI labels list, if None, use the whole brain
        snr, controls lambda
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
    # regression, return coefficients and F-values 
    p = X.shape[1] 
    [dim0,dim1,dim2] = roi_data.shape
    coef = np.zeros([dim0,dim1,p])
    F = np.zeros([dim0,dim1]) if Flag_reg_stats else None
    linreg_op = np.dot(la.inv(X.T.dot(X)),X.T)
    for i in range(dim0):
        for j in range(dim1):
            tmpY = roi_data[i,j,:]
            tmp_coef = linreg_op.dot(tmpY)
            coef[i,j,:] = tmp_coef
            if Flag_reg_stats:
                tmpY_hat = np.dot(X,tmp_coef)
                tmp_res = tmpY_hat-tmpY
                SSE = np.dot(tmp_res,tmp_res)
                SST =  np.sum((tmpY-np.mean(tmpY))**2)
                if SSE== 0:
                    F[i,j] += 0
                else:
                    F[i,j] += (SST-SSE)/(p-1)/(SSE/(n_trials-p))
                    
                    
    result_dict = dict(coef = coef, F = F, sel = sel,roi_data_3D = roi_data)
    return result_dict
#===============================================================
def get_MSE_mne_regression(evoked_list, fwd, X, coef, labels):
    '''
    Use the mne regression coefficients to get predicted sensor data, 
    then abtain the mean of squared error
    Input:
        evoked_list, a list of evoked objects
        fwd, the forward solution
        X, the design matrix,
        coef, the regression coefficients, [n_sources, ntimes, p] real
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
    n_trials,p = X.shape
    if n_trials != len(evoked_list):
        raise ValueError("the numbers of trials do not match")   
    SSE = 0.0
    for r in range(n_trials):
        # STFT coefficients of current trial
        predicted_ts = np.sum(coef*X[r,:],axis = 2)
        predicted_sensor = G.dot(predicted_ts)
        SSE += np.sum((evoked_list[r].data - predicted_sensor)**2)
    MSE = SSE/(n_trials)
    return MSE       


# ==============================================================
def select_lambda_tuning_mne_regression_cv(evoked_list, inverse_operator,
                                                fwd, X,  cv_partition_ind,
                                                snr_tuning_seq, 
                                                labels = None):
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
            tmp_result = mne_regression(evoked_list_train, inverse_operator, 
                                     Xtrain,  labels = labels,
                                     snr=tmp_snr)
            coef = tmp_result['coef']
            # Now do the prediction
            tmp_MSE = get_MSE_mne_regression(evoked_list_test, fwd, Xtest,
                                                  coef,  labels = labels)
            cv_MSE[i,j] = tmp_MSE 
    cv_MSE = cv_MSE.mean(axis = 1) 
    best_ind = np.argmin(cv_MSE)
    snr_tuning_star = snr_tuning_seq[best_ind]
    return snr_tuning_star, cv_MSE
           
                                    
    