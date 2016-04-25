# -*- coding: utf-8 -*-
"""
@author: yingyang
"""

import numpy as np
import scipy
from copy import deepcopy
import mne 
mne.set_log_level('warning')
from mne.inverse_sparse.mxne_optim import  _PhiT
import os
from MNE_stft.mne_stft_regression_individual_G import (mne_stft_regression_individual_G, 
                        select_lambda_tuning_mne_stft_regression_cv_individual_G)                     
import STFT_R                   
#==============================================================================
def get_solution_individual_G(evoked_list, fwd_list, G_ind, noise_cov, X,
                              labels = None, 
                 alpha_seq = None, beta_seq = None, gamma_seq =None, delta_seq = None,
                 snr_tuning_seq = None,
                 active_set = None,coef_non_zero_mat = None,
                 wsize = 16, tstep = 4,
                 maxit = 20, tol = 1e-2,
                 method = "STFT-R",
                 L2_option = 1, 
                 ROI_weight_param = 0,
                 Maxit_J=10, Incre_Group_Numb=50, dual_tol=0.1,
                 depth = None, 
                 Flag_backtrack = False, L0 = 1.0, eta = 1.1,
                 Flag_verbose = False,
                 Z0_L2 = None, n_active_ini = 200):
    '''
    Get the mne solutions or the solution by my method, "stft_reg"
    Can only be applied to fixed orientation, i.e. each source point has only one direction. 
    Input:
        evoked_list, list of evoked objects,  [n_trials,]
        fwd_list, a list distinct forward solutions, when the trials are organized in runs, 
                different runs may have different forward solutions due to head movement
        G_ind, [n_trials,] indices of runs, showing which fwd solution for each trial to use
        X, [n_trials,p] the design matrix
        noise_cov, noise_covariance matrix object by MNE
        labels, [n_ROI,] a list of MNE label objects, describing each ROI,
                can be None, if yes, each individual source point is a group
        alpha_seq, beta_seq, gamma_seq, the tuning parameters for group lasso only for methods "STFT-R"
        delta_seq, tuning parameters for the L2 solution for "STFT-R" only
        snr_tuning_seq, sequence of the "snr" parameters for MNE, 
                      L2 tuning parameters for MNE = 1/snr**2, 
        active_set, only useful when L2_option == 2, indices of active sources, 
                  [n_dipoles,]  boolean
        coef_non_zero_mat, only useful when L2_option == 2, 
                    this boolean matrix is the active entries of the regression coefficients
        wsize, tstep, window size and time step size of the STFT transform
        maxit, maximum number of iterations in the optimization
        tol, stoppting criterion, when the relative change of the objective function is smaller than tol, stop
        method = "MNE-R" or "STFT-R" 
        L2_option, option for the L2 solution for method "STFT-R" only,
            if 0, only get the L21 solution
            if 1, get the L21 solution first, then obtain an L2 solution constrained on the non-zero elements of the L21 solution
            if 2, only do L2 solution, using the non-zero elements in "active_set" and "coef_non_zero_mat"
        ROI_weight_param:  only for method "STFT-R", a ratio of first level penalty weights  within ROIs / outside ROIs
                       Group weights = 1/sources in the group multiplies by this ratio,
                       then normalized
                       if 0, there is no penalty on the ROIs               
        Maxit_J,  only for method "STFT-R", maximum number of greedy steps to take in the active-set gready method  
        Incre_Group_Numb: only for method "STFT-R", in the greedy step, each time include this number of first-level groups
        dual_tol: if the violation of KKT for the greedy method is smaller than this value, stop
        depth, 0 to 1, the depth prior defined in the MNE algorithm, it normalizes the forward matrix, 
               by dividing each column with  (np.sum(G**2, axis = 0))**depth, such that deeper source points can 
               larger influence. 
        Flag_verbose = False, only for method "STFT-R", whether to print the optimization details of solving L21.               
    Output:
        result_dict = dict(Z= Z, active_set = active_set,
                           active_t_ind = active_t_ind,
                           coef_non_zero_mat = coef_non_zero_mat,
                           source_data = source_data,
                           method = method,
                           snr_tuning_star = snr_tuning_star,
                           alpha_star = alpha_star, 
                           beta_star = beta_star,
                           gamma_star = gamma_star,
                           delta_star = delta_star)
        ['Z'] [active_set.sum(), n_coefs*p]
        ['active_set'], [n_dipoles,] boolean
        ['active_t_ind'], active_t_ind here may not match the size of Z, 
                  since Z is already expanded,
                  but active_t_ind is useful to compute the bootstrap results
        ['coef_non_zero_mat'], [active_set.sum(), n_coefs*pq], boolean matrix,
                  a more detailed active set
        ['source_data'], [active_set.sum(), n_times, n_trials]
        ['method']
        and all other tuning paramters
    '''
    #if not mne.forward.is_fixed_orient(fwd_list[0]):
    #    raise ValueError('Forward solution does not has fixed orientation')
    n_channels, n_times = evoked_list[0].data.shape
    n_trials = len(evoked_list)
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq
    p = X.shape[1]
    n_dipoles = fwd_list[0]['src'][0]['nuse'] + fwd_list[0]['src'][1]['nuse']
    # for mne depth = None!, for STFT-R depth = 0.8
    # default depth = none
    # depth = None
    cv_partition_ind = np.zeros(n_trials)
    cv_partition_ind[1::2] = 1
    n_run = len(np.unique(G_ind))
                                                  
    if method == "MNE-R":
        inverse_operator_list = list()
        for run_id in range(n_run):
            inverse_operator = mne.minimum_norm.make_inverse_operator(evoked_list[0].info, 
                                                              fwd_list[run_id], noise_cov, depth = depth,
                                                              fixed = True) 
            inverse_operator_list.append(inverse_operator)   
        # create an inverse operator first
        # tuning parameter range
        if snr_tuning_seq is None:                                                    
            snr_tuning_seq = np.array([0.5,1,2,3])
        if len(snr_tuning_seq) == 1:
            snr_tuning_star = snr_tuning_seq[0]
            cv_MSE_mne = 0.0
        else:
            snr_tuning_star, cv_MSE_mne = select_lambda_tuning_mne_stft_regression_cv_individual_G\
                         (evoked_list, inverse_operator_list,fwd_list, G_ind, X, cv_partition_ind,
                          snr_tuning_seq,labels = None,wsize=wsize, tstep = tstep)
        
        mne_result = mne_stft_regression_individual_G(evoked_list, inverse_operator_list, G_ind, X, 
                                   labels = None, snr = snr_tuning_star,
                                   wsize = wsize, tstep = tstep, method = "MNE")
        Z_coef_mne = mne_result['coef']
        Z = np.zeros([n_dipoles, p*n_coefs], dtype = np.complex)  
        for i in range(n_dipoles):
            for l in range(p):
                Z[i,l*n_coefs:(l+1)*n_coefs] \
                  = np.ravel(Z_coef_mne[i,:,:,l], order = 'C') 
        # return the solutions
        active_set = np.ones(n_dipoles, dtype = np.bool)
        active_t_ind = np.ones(n_step, dtype = np.bool)
        coef_non_zero_mat = np.abs(Z)>0
        alpha_star, beta_star, gamma_star, delta_star = 0,0,0,0
        cv_MSE_lasso, cv_MSE_L2= 0,0
        
    elif method == "STFT-R":
        if alpha_seq is None:
            alpha_seq = np.exp(np.arange(6,4,-1))*10.0
        if beta_seq is None:
            beta_seq = np.exp(np.arange(3,0,-1))*2.0
        if gamma_seq is None:
            gamma_seq = np.exp(np.arange(0,-1,-1))
        if delta_seq is None:
            delta_seq = np.exp(np.arange(-6,2,1))    
                
        if L2_option == 0 or L2_option == 1:
            # use the last run as initial value?
            inverse_operator1 = mne.minimum_norm.make_inverse_operator(evoked_list[0].info, 
                                                              fwd_list[-1], noise_cov, depth = None,
                                                              fixed = True) 
            stc_mne = mne.minimum_norm.apply_inverse(evoked_list[-1], inverse_operator1, 
                                             lambda2=1.0,method='MNE')                   
            mne_val = np.mean(np.abs(stc_mne.data), axis = 1)
            mne_val_ind = np.argsort(-1*mne_val)
            active_set_z0 = np.zeros(n_dipoles, dtype = np.bool)
            active_set_z0[mne_val_ind[0:n_active_ini]] = True
            print "selecting the largest %d dipoles" %n_active_ini
        else:
            active_set_z0 = active_set.copy()
        if depth is None:
            depth = 0.0
        # Note before Oct 13, depth was set to 0.8
        Z, active_set, active_t_ind, alpha_star, beta_star, gamma_star, delta_star,\
           cv_MSE_lasso, cv_MSE_L2 = STFT_R.get_STFT_R_for_epoch_data.get_STFT_R_solution(evoked_list,X, fwd_list, G_ind, noise_cov,
                                labels,  ROI_weight_param,
                                active_set_z0, 
                                alpha_seq,beta_seq,gamma_seq,
                                loose= None, depth=depth, maxit=maxit, tol=tol,
                                wsize=wsize, tstep=tstep, window=0.02,
                                L2_option = L2_option, delta_seq = delta_seq,
                                coef_non_zero_mat = coef_non_zero_mat, Z0_l2 = Z0_L2,
                                Maxit_J=Maxit_J, Incre_Group_Numb=Incre_Group_Numb, 
                                dual_tol= dual_tol,
                                Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta,
                                Flag_verbose = Flag_verbose,
                                Flag_nonROI_L2 = False)
        coef_non_zero_mat = np.abs(Z)>0
        snr_tuning_star = 0
        cv_MSE_mne = 0
    # common out put
    # return the solutions
    result_dict = dict(Z= Z, active_set = active_set,
                       active_t_ind = active_t_ind,
                       coef_non_zero_mat = coef_non_zero_mat,
                       method = method,
                       snr_tuning_star = snr_tuning_star,
                       alpha_star = alpha_star, 
                       beta_star = beta_star, 
                       gamma_star = gamma_star,
                       delta_star = delta_star,
                       cv_MSE_lasso = cv_MSE_lasso,
                       cv_MSE_L2 = cv_MSE_L2, 
                       cv_MSE_mne = cv_MSE_mne)      
    return result_dict
# =============================================================================
def compare_with_truth_individual_G(Z, active_set, source_data, 
                       Z_true, true_stc_data_list,  true_ind, 
                       ROI_ind, n_coefs, X, label_ind,
                       method,  
                       Flag_reconstruct_source = False,
                       tstep = 4, wsize = 16, Flag_sparse = True):
    '''
    Given a stft coefficient solution, compare with the truth
    Input:
        Z, [active_set.sum(), n_coefs*p]
        active_set, [n_dipole,] boolean
        source_data, if None, use Z to estimate it, if given,
                     it should be [active_set.sum(),n_times, n_trials]
                     The source_data outputted by the model may not be the same as predicted by Z,
                     for mne, it is the the raw mne solution before reconstruction, 
                     for stft_reg, it could be the trial by trial model
        Z_true, [n_dipoles, n_coefs*p]
        true_stc_data_list, list of the true stc data
        true_ind, indices of truly active dipoles
        ROI_ind, indices of dipoles that we care about
        n_coefs,
        X, [n_trials, p]
        label_ind, list of indices in each ROI
        ROI_curve,  PCA of the ROI curve
        ROI_curve_var_percent, percent of variance the components explain
        method = "stft_reg", "mne" or "mne_stft"
        Flag_reconstruct_source = False,
            if True, use Z to construct the source data
        
    Ouput:
        result = dict(coef_error_ROI = coef_error_ROI, coef_error = coef_error,
                  curve_error_ROI = curve_error_ROI, curve_error = curve_error,
                  MSE_source = MSE_source, MSE_source_ROI = MSE_source_ROI)
    ''' 
    
    # ==================== the error in source space =======================
    # if source_data is None, do not compute the MSEs
    n_dipoles = active_set.shape[0]
    [n_trials, p] = X.shape
    MSE_source_ROI, MSE_source = 0.0,0.0
    MSE_source_abs_ROI, MSE_source_abs = 0.0,0.0
    if source_data is not None:
        n_times = source_data.shape[1]
        # compute the distance to truth with the given source_data
        for i in range(n_trials):
            # estimated data, full and within ROI
            tmp_stc_full = np.zeros([n_dipoles, n_times])
            tmp_stc_full[active_set, :] = source_data[:,:,i]
            tmp_stc_ROI = tmp_stc_full[ROI_ind,:]
            # true data
            if Flag_sparse: # the true data contains only dipoles in true_ind
                tmp_true_stc_data = np.zeros([n_dipoles,n_times])
                tmp_true_stc_data[true_ind,:] = true_stc_data_list[i]
            else:  # the true data contains all dipoles
                tmp_true_stc_data = true_stc_data_list[i]
                
            tmp_true_stc_data_ROI = tmp_true_stc_data[ROI_ind,:]
            
            MSE_source_ROI += np.sum(( tmp_stc_ROI - tmp_true_stc_data_ROI )**2)
            MSE_source += np.sum(( tmp_stc_full - tmp_true_stc_data )**2)
            MSE_source_abs_ROI += np.sum(( np.abs(tmp_stc_ROI) - np.abs(tmp_true_stc_data_ROI) )**2)
            MSE_source_abs += np.sum(( np.abs(tmp_stc_full) - np.abs(tmp_true_stc_data) )**2)
            
        MSE_source_ROI /= n_trials
        MSE_source /= n_trials
        MSE_source_abs_ROI /= n_trials
        MSE_source_abs /= n_trials
        
    # ==================== the reconstruction error in source space =======================
    MSE_source_ROI_recon, MSE_source_recon = 0.0,0.0
    MSE_source_abs_ROI_recon, MSE_source_abs_recon = 0.0,0.0
    
    if Flag_reconstruct_source:
        n_times = true_stc_data_list[0].shape[1]
        n_freq = wsize//2+1
        n_step = n_times // tstep
        phiT = _PhiT(tstep, n_freq, n_step, n_times)
        for i in range(n_trials):
            if method == "mne_stft" or method == "stft_reg":
                tmpZ = np.reshape(Z,[active_set.sum(), p, n_coefs])
                tmpZ = np.swapaxes(tmpZ, 1,2)
                tmpZ = np.sum(tmpZ*X[i,:],axis = 2)
                tmp_source_data = phiT(tmpZ)
            else:
                # method = mne, Z [n_sources, n_times, p]
                tmp_source_data = np.sum(Z*X[i,:],axis = 2)
                
            tmp_stc_full = np.zeros([n_dipoles, n_times])
            tmp_stc_full[active_set, :] = tmp_source_data
            tmp_stc_ROI = tmp_stc_full[ROI_ind,:]
            
            # the true data
            if Flag_sparse:
                tmp_true_stc_data = np.zeros([n_dipoles,n_times])
                tmp_true_stc_data[true_ind,:] = true_stc_data_list[i]
            else:
                tmp_true_stc_data = true_stc_data_list[i]
        
            true_stc_data_ROI = tmp_true_stc_data[ROI_ind,:]
            MSE_source_ROI_recon += np.sum((tmp_stc_ROI - true_stc_data_ROI)**2)
            MSE_source_recon += np.sum((tmp_stc_full - tmp_true_stc_data)**2)
            MSE_source_abs_ROI_recon += np.sum( (np.abs(tmp_stc_ROI) - np.abs(true_stc_data_ROI) )**2)
            MSE_source_abs_recon += np.sum(( np.abs(tmp_stc_full) - np.abs(tmp_true_stc_data))**2)
        
        MSE_source_ROI_recon /= n_trials
        MSE_source_recon /= n_trials
        MSE_source_abs_ROI_recon /= n_trials
        MSE_source_abs_recon /= n_trials
    
    # ======================other errors ===================================    
    
    # ============errors of coefficients================================
    Z_full = np.zeros([n_dipoles,n_coefs*p], dtype = np.complex)
    Z_full[active_set,:] = Z
    Z_ROI = Z_full[ROI_ind,:]
    
    # Z_true is sparse, only at the ROIs
    Z_true_full = np.zeros([n_dipoles,n_coefs*p], dtype = np.complex)
    Z_true_full[true_ind,:] = Z_true
    Z_true_ROI = Z_true_full[ROI_ind,:]
    
    coef_error_ROI = np.sqrt( np.sum( np.abs( Z_ROI - Z_true_ROI ) **2) )
    coef_error = np.sqrt( np.sum( np.abs( Z_full - Z_true_full ) **2) )
    
    # measure the correlation 
    result = dict(coef_error_ROI = coef_error_ROI,coef_error = coef_error,
                  MSE_source = MSE_source, MSE_source_ROI = MSE_source_ROI,
                  MSE_source_ROI_recon = MSE_source_ROI_recon, 
                  MSE_source_recon = MSE_source_recon,
                  MSE_source_abs_ROI_recon = MSE_source_abs_ROI_recon,
                  MSE_source_abs_recon = MSE_source_abs_recon,
                  MSE_source_abs_ROI = MSE_source_abs_ROI,
                  MSE_source_abs = MSE_source_abs)
    return result                                  
             
# ========================== predicted error on the held-out data =============                               
def get_sensor_MSE_on_held_data_individual_G(Z, active_set, test_evoked_list,
                       test_X, G_list, G_ind_test, method, 
                       wsize = 16, tstep = 4):
    '''
    Compute the mean squared error on the held-out sensor data.
    This is done on the original sensor data, before pre-whitening and SSP
    To make sure this is comparable for all three method, I rewrote the computation here. 
    Input: 
        Z, [active_set.sum(), n_coef*p]
        active_set,
        test_evoked_list, evoked list for testing
        test_X, [n_trials, p]
        G_list, 
        G_ind_test
        method = "STFT-R" "MNE-R"
        wsize, tstep
    Output: MSE_sensor    
    '''
    # note that M is not whitened here
    n_channels, n_times = test_evoked_list[0].data.shape
    n_trials = len(test_evoked_list)
    p = test_X.shape[1]
    
    if method == "MNE-R":
        n_times = test_evoked_list[0].data.shape[1]
        n_step = int(np.ceil(n_times/float(tstep)))
        n_freq = wsize//2+1
        n_step = n_times // tstep
        n_coefs = n_freq*n_step
        phiT = _PhiT(tstep, n_freq, n_step, n_times)
        
    
    M = np.zeros([n_channels, n_times, n_trials ])
    for i in range(n_trials):
         M[:,:,i] = test_evoked_list[i].data 
 
    MSE_sensor = 0.0
    for i in range(n_trials):
        tmp_Z = np.reshape(Z, [active_set.sum(), p, n_coefs])
        tmp_Z = np.sum(  np.swapaxes(tmp_Z, 1,2) * test_X[i,:], axis = 2)
        tmp_source = phiT(tmp_Z)
        predicted = G_list[G_ind_test[i]][:,active_set].dot(tmp_source)     
        MSE_sensor +=  np.sum ( (M[:,:,i] - predicted)**2 )
        
    MSE_sensor /= n_trials 
    return MSE_sensor  

#==============================================================================
def get_bootstrapped_solution_individual_G(evoked_list,fwd_list, G_ind, noise_cov, X,Z, active_set, 
                              coef_non_zero_mat,path,fname,
                       labels, label_ind, 
                       method = "STFT-R",
                       maxit = 50, tol = 1e-2, 
                       B = 50, wsize = 16, tstep = 4, Rescale_Flag = True,
                       delta_seq = None, snr_tuning_seq = None,
                       depth = None):
    '''
    Compute bootstrapped samples of the regression coefficients. 
    if method = "STFT-R", only use L2 method on the given non-zero elements

    Input: 
        evoked_list, list of evoked objects,  [n_trials,]
        fwd_list, a list distinct forward solutions, when the trials are organized in runs, 
                different runs may have different forward solutions due to head movement
        G_ind, [n_trials,] indices of runs, showing which fwd solution for each trial to use
        noise_cov, noise_covariance matrix object by MNE
        X, [n_trials,p] the design matrix
        Z, the estimated complex regression coefficients, the bootstrapped sample will be computed based on it
        active_set, only useful for  method "STFT-R", indices of active sources, 
                  [n_dipoles,]  boolean
        coef_non_zero_mat,  only useful for  method "STFT-R"
                    this boolean matrix is the active entries of the regression coefficients
        path, director to temporarily store the bootstrapped data
        fname, names of the bootstrapped data           
        labels, [n_ROI,] a list of MNE label objects, describing each ROI
        label_ind, list of  source indices of each ROI,
        maxit, tol, only for method "STFT-R", stopping criteria of the L2 optimization 
        B, number of bootstrapped samples
        wsize, tstep, window size and time step size of the STFT transform
        Rescale_Flag, if True, rescale the error when generating boostrapped samples
        delta_seq, tuning parameters for the L2 solution for "STFT-R" only
        snr_tuning_seq, sequence of the "snr" parameters for MNE, 
                      L2 tuning parameters for MNE = 1/snr**2, 
        depth, 0 to 1, the depth prior defined in the MNE algorithm, it normalizes the forward matrix, 
               by dividing each column with  (np.sum(G**2, axis = 0))**depth, such that deeper source points can 
               larger influence.                

    Output:
         btstrp_result = dict(se_absZ = se_absZ, Z_btstrp = Z_btstrp,
         L2_tuning_par = L2_tuning_par)    
    '''
         
         
    # generate the bootstrapped data
    # Z is a full matrix, covering all columns, so active_t_ind is all True
    n_channels, n_times = evoked_list[0].data.shape
    n_trials = len(evoked_list)
    M = np.zeros([n_channels, n_times, n_trials ])
    for i in range(n_trials):
        M[:,:,i] = evoked_list[i].data  
    
    n_run = len(np.unique(G_ind))
    G_list = list()
    for run_id in range(n_run):
        G_list.append(fwd_list[run_id]['sol']['data'])
   
    n_step = int(np.ceil(n_times/float(tstep)))
    STFT_R.get_btstrp_sample.get_btstrp_stft_regresion_with_active_set \
                                  (M,G_list, G_ind,X, 
                                   Z, active_set, np.ones(n_step,dtype=np.bool),
                                   coef_non_zero_mat,
                                   path, fname+"_"+method,
                                   B = B, wsize=wsize, tstep = tstep, 
                                   Rescale_Flag = Rescale_Flag) 
                                   
    btstrp_result_list = list()  
    L2_tuning_par = np.zeros(B)
    for i in range(B):
        mat_name = path + fname+"_"+method + "_btstrp_%04i.mat" %i 
        tmp_dict = scipy.io.loadmat(mat_name)
        btstrp_M = tmp_dict['btstrp_M']   
        # create a new evoked list for the bootstrapped data
        evoked_list_btstrp = deepcopy(evoked_list)
        for j in range(n_trials):
            evoked_list_btstrp[j].data = btstrp_M[:,:,j] 
        tmp_result = get_solution_individual_G(evoked_list_btstrp, fwd_list,G_ind, noise_cov, X, 
                 labels, 
                 alpha_seq = None, beta_seq=None, gamma_seq = None,
                 delta_seq = delta_seq, 
                 snr_tuning_seq = snr_tuning_seq,
                 active_set = active_set, coef_non_zero_mat = coef_non_zero_mat,
                 wsize = wsize, tstep = tstep,
                 maxit = maxit, tol = tol,
                 method = method,
                 L2_option = 2, depth = depth)   
                        
        btstrp_result_list.append(tmp_result)
        if method == "STFT-R":
            L2_tuning_par[i] = tmp_result['delta_star']
        else:
            L2_tuning_par[i] = tmp_result['snr_tuning_star']
        # Finally remove the btstrp file from the directory 
        os.remove(mat_name)
    os.remove(path + fname+"_"+method + "_btstrp_params.mat") 
    
    # save the result in a big matrix
    Z_btstrp = np.zeros(np.hstack([btstrp_result_list[0]['Z'].shape,B]), dtype = np.complex)
    for i in range(B):
        Z_btstrp[:,:,i] = btstrp_result_list[i]['Z']
                                        
    # return the confidence intervals? 
    se_absZ = np.std(np.abs(Z_btstrp),axis = 2)
    btstrp_result = dict(se_absZ = se_absZ, Z_btstrp = Z_btstrp,
                         L2_tuning_par = L2_tuning_par)    
    return btstrp_result