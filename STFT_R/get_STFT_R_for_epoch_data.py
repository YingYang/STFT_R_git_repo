# -*- coding: utf-8 -*-
"""
Changes: 20160216
(1) source_weighting changed from MNE v0.8 to MNE v0.11, 
  v0.11    G_tilde = G*source_weighting   Z = Z_tilde* source_weighting
  v0.8     G_tilde = G/source_weighting   Z = Z_tilde/source_weighting
  Now I use their "_reapply_source_weighting" function
(2) label_list can be None now, in this case, alpha penalty is applied 
   on each dipole
(3) backtracking added to both L21 and L2 optimization, 
(4) active_t_ind, the active set of time steps was removed in L21, previously, 
    it was not useful either, it is always set to True in every step.     
"""

from copy import deepcopy
import numpy as np

import mne
from mne.source_estimate import SourceEstimate
from mne.minimum_norm.inverse import _prepare_forward
from mne.forward import compute_orient_prior, is_fixed_orient, _to_fixed_ori
from mne.inverse_sparse.mxne_optim import mixed_norm_solver, norm_l2inf, tf_mixed_norm_solver
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT 
from mne.inverse_sparse.mxne_inverse import _reapply_source_weighting
from mne.utils import logger, verbose
from mne.inverse_sparse.mxne_inverse import _prepare_gain,_make_sparse_stc, _window_evoked
from mne.source_space import label_src_vertno_sel

from . import L21_active_set as L21solver                                
from . import L2_tsparse  as L2solver 

def get_STFT_R_solution(evoked_list,X, fwd_list0, G_ind, noise_cov,
                                label_list,  GroupWeight_Param,
                                active_set_z0, 
                                alpha_seq,beta_seq,gamma_seq,
                                loose= None, depth=0.0, maxit=500, tol=1e-4,
                                wsize=16, tstep=4, window=0.02,
                                L2_option = 0, delta_seq = None,
                                coef_non_zero_mat = None, Z0_l2 = None,
                                Maxit_J=10, Incre_Group_Numb=50, dual_tol=0.01,
                                Flag_backtrack = True, L0 = 1.0, eta = 1.5,
                                Flag_verbose = False,
                                Flag_nonROI_L2 = False):
    '''
    Compute the L21 or L2 inverse solution of the stft regression. 
    If Flag_trial_by_trial == True, use the "trial-by-trial" model for estiamtion,
    otherwise, use the simpler model without trial by trial terms
    Input:
        evoked_list, a list of evoked objects
        X, [n_trials, p] design matrix of the regresison
        fwd_list0, a list of n_run  forward solution object
        run_ind, [n_trials, ] run index, starting from zero
        noise_cov, the noise covariance matrix
        label_list, a list of labels or ROIs. 
                    it can be None, in that case, each individual dipole is
                    one group, also, GroupWeight_Param becomes invalid, 
                    penalty alpha is applied to every dipole, Flag_nonROI_L2 is
                    set to False too. 
        GroupWeight_param, a ratio of weights  within ROIs / outside ROIs
                       Group weights = 1/ n_dipoles in the group, times ratio,
                       then normalized
        active_set_z0, the initial active_set
        alpha_seq, tuning sequence for alpha, (the group penalty)
        beta_seq, tuning sequence for beta, ( penalty for a single STFT basis function )
        loose, depth, the loose and depth paramter for the source space
        maxit, the maximum number of iteration
        tol, numerical tolerance of the optimizaiton
        wsize, window size of the STFT 
        tstep, time steps of the STFT
        window, windowing of the data, just to remove edge effects
        L2_option, 0, only compute the L21 solution
                   1, after computing the L21 solution, 
                       use them as the active set and get an L2 solution.
                       If delta_seq is provided, run cross validation 
                                    to get the best tuning parameter.
                   2, only compute the L2 solution, 
                      coef_non_zero_mat must not be None for this option,
                      active_set_z0, active_t_ind must correspond to the active set
        delta_seq, the tuning sequence for the L2 solution
                   if None, a default value will be used. 
        coef_non_zero_mat, [active_set.sum(), n_coefs*p], boolean matrix, active set
            e.g. coef_non_zero_mat = np.abs(Z)>0
        Z0_l2, the same size as coef_non_zero_mat, the initial value for L2 problems
        verbose, mne-python parameter, level of verbose
        Flag_nonROI_L2 = False, if true, all dipoles outside the ROIs are one large group. 
        Maxit_J, when solving the L21 problem, maximum number of greedy steps to take in the active-set gready method  
        Incre_Group_Numb: when solving the L21 problem, in the greedy step, each time include this number of first-level groups
        dual_tol: when solving the L21 problem,, if the violation of KKT for the greedy method is smaller than this value, stop
        depth, 0 to 1, the depth prior defined in the MNE algorithm, it normalizes the forward matrix, 
               by dividing each column with  (np.sum(G**2, axis = 0))**depth, such that deeper source points can 
               larger influence.
               To make it valid, the input forward objects must not have fixed orientation!
        Flag_verbose,   whether to print the optimization details of solving L21.  
        Flag_backtrack = True, L0 = 1.0, eta = 1.5,  parameters for backtracking          
       
    Output:
        Z_full, [n_dipoles, n_coefs*p], complex matrix, the regression results
        active_set, [n_dipoles,] boolean array, dipole active set
        active_t_ind, [n_step,], boolean array, temporal active set, should be a full True vector
        stc_list, a list of stc objects, the source solutions
        alpha_star, the best alpha
        beta_star, the best beta
        gamma_star, the best gamma
        delta_star, the best delta
    '''
    # =========================================================================    
    # some parameters to prepare the forward solution
    weights, weights_min, pca=None, None, True 
    all_ch_names = evoked_list[0].ch_names
    info = evoked_list[0].info
    n_trials = len(evoked_list)
    # put the forward solution in fixed orientation if it's not already
    n_runs = len(np.unique(G_ind))
    G_list = list()
    whitener_list = list()
    fwd_list = deepcopy(fwd_list0)
    for run_id in range(n_runs):
        if loose is None and not is_fixed_orient(fwd_list[run_id]):
            # follow the tf_mixed_norm
            _to_fixed_ori(fwd_list[run_id])
        
        # mask should be None
        gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
             fwd_list[run_id], info, noise_cov, pca, depth, loose, weights, weights_min)                                                                         
        G_list.append(gain)
        whitener_list.append(whitener)                                    
    # to debug
    # print np.linalg.norm(G_list[0]-G_list[1])/np.linalg.norm(G_list[0])
    # print np.linalg.norm(whitener_list[0]-whitener_list[1])
    # the whitener is the same across runs
    # apply the window to the data
    if window is not None:
        for r in range(n_trials):
            evoked_list[r] = _window_evoked(evoked_list[r], window)
    # prepare the sensor data
    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    _, n_times = evoked_list[0].data[sel].shape
    n_sensors = G_list[0].shape[0]
    
    M = np.zeros([n_sensors, n_times, n_trials], dtype = np.float)
    # Whiten data
    logger.info('Accessing and Whitening data matrix.')
    # deal with SSP
    # the projector information should be applied to Y
    info = evoked_list[0].info
    # all forward solutions must hav ethe same channels, 
    # if there are bad channels, make sure to remove them for all trials before using this function
    fwd_ch_names = [c['ch_name'] for c in fwd_list[0]['info']['chs']]
    ch_names = [c['ch_name'] for c in info['chs']
                if (c['ch_name'] not in info['bads']
                    and c['ch_name'] not in noise_cov['bads'])
                and (c['ch_name'] in fwd_ch_names
                     and c['ch_name'] in noise_cov.ch_names)]
    # ?? There is no projection in the 0.11 version, should I remove this too
    # proj should be None, since the projection should be applied after epoching
    proj, _, _ = mne.io.proj.make_projector(info['projs'], ch_names)
    for r in range(n_trials):
        M[:,:,r] = reduce(np.dot,[whitener,proj, evoked_list[r].data[sel]])
    #=========================================================================
    # Create group information
    src = fwd_list[0]['src']
    n_dip_per_pos = 1 if is_fixed_orient(fwd_list[0]) else 3
    # number of actual nodes, each node can be associated with 3 dipoles
    n_dipoles = G_list[0].shape[1]//n_dip_per_pos
    ## this function is only for n_dip_per_pos == 1
    #if n_dip_per_pos != 1:
    #    raise ValueError("n_orientation must be 1 for now!")
    ##
    if label_list is None:
        nROI = 0
        Flag_nonROI_L2 = False
    else:
        label_ind = list()
        for label in label_list:
            # get the column index corresponding to the ROI
            _, tmp_sel = label_src_vertno_sel(label,src)                                       
            label_ind.append(tmp_sel) 
        nROI = len(label_ind)
                                      
    DipoleGroup = list()
    isinROI = np.zeros(n_dipoles, dtype = np.bool)
    if n_dip_per_pos == 1:
        for i in range(nROI):
            DipoleGroup.append((np.array(label_ind[i])).astype(np.int)) 
            isinROI[label_ind[i]] = True
        # dipoles outside the ROIs
        notinROI_ind = np.nonzero(isinROI==0)[0]
        if Flag_nonROI_L2:
            DipoleGroup.append(notinROI_ind.astype(np.int))           
        else:
            for i in range(len(notinROI_ind)):
                DipoleGroup.append(np.array([notinROI_ind[i]]))
    else:
        for i in range(nROI):
            tmp_ind = np.array(label_ind[i])
            tmp_ind = np.hstack([tmp_ind*3,
                             tmp_ind*3+1, 
                             tmp_ind*3+2])
            DipoleGroup.append(tmp_ind.astype(np.int)) 
            isinROI[tmp_ind] = True
        # dipoles outside the ROIs
        notinROI_ind = np.nonzero(isinROI==0)[0]
        if Flag_nonROI_L2:
            DipoleGroup.append(notinROI_ind.astype(np.int))     
        else:   
            for i in range(len(notinROI_ind)):
                DipoleGroup.append(np.array([3*notinROI_ind[i], 
                                             3*notinROI_ind[i]+1,
                                             3*notinROI_ind[i]+2]).astype(np.int))  
    # Group weights, weighted by number of dipoles in the group  
    DipoleGroupWeight = 1.0/np.array([len(x) for x in DipoleGroup ])
    DipoleGroupWeight[0:nROI] *= GroupWeight_Param
    DipoleGroupWeight /= DipoleGroupWeight.sum()
        
    # =========================================================================
    # STFT constants
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq
    p = X.shape[1]
    # =========================================================================
    # Scaling to make setting of alpha easy, modified from tf_mixed_norm in v0.11
    alpha_max = norm_l2inf(np.dot(G_list[0].T, M[:,:,0]), 
                           n_dip_per_pos, copy=False)
    alpha_max *= 0.01
    for run_id in range(n_runs):
        G_list[run_id] /= alpha_max
    # mne v0.11  tf_mixed_norm,  "gain /= alpha_max    source_weighting /= alpha_max"
    # so maybe the physcial meaning of source_weighting changed to its inverse
    # i.e. G_tilde = G*source_weighting
    # for MNE0.8, I used
    #source_weighting *= alpha_max 
    source_weighting /= alpha_max
    cv_partition_ind = np.zeros(n_trials)
    cv_partition_ind[1::2] = 1
    cv_MSE_lasso, cv_MSE_L2 = 0,0
    # =========================================================================
    if L2_option == 0 or L2_option == 1: 
        #  compute the L21 solution
        # setting the initial values, make sure ROIs are in the initial active set
        isinROI_ind = np.nonzero(isinROI)[0]
        if n_dip_per_pos == 1:
            active_set_z0[isinROI_ind] = True
        else:
            active_set_z0[3*isinROI_ind ] = True
            active_set_z0[3*isinROI_ind+1] = True
            active_set_z0[3*isinROI_ind+2] = True
            
        active_set_J_ini = np.zeros(len(DipoleGroup), dtype = np.bool)
        for l in range(len(DipoleGroup)):
            if np.sum(active_set_z0[DipoleGroup[l]]) > 0:
                active_set_J_ini[l] = True
        # if alpha and beta are sequences, use cross validation to select the best
        if len(alpha_seq) > 1 or len(beta_seq) > 1 or len(gamma_seq) >1:
            print "select alpha,beta and gamma"
            alpha_star, beta_star, gamma_star, cv_MSE_lasso = L21solver.select_alpha_beta_gamma_stft_tree_group_cv_active_set(
                                         M,G_list, G_ind, X,
                                         active_set_J_ini, 
                                         DipoleGroup,DipoleGroupWeight,
                                         alpha_seq, beta_seq, gamma_seq, cv_partition_ind,
                                         n_orient=n_dip_per_pos, 
                                         wsize=wsize, tstep = tstep, 
                                         maxit=maxit, tol = tol,
                                         Maxit_J = Maxit_J, Incre_Group_Numb = Incre_Group_Numb,
                                         dual_tol = dual_tol,
                                         Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta,
                                         Flag_verbose=Flag_verbose)
        else:
            alpha_star, beta_star, gamma_star = alpha_seq[0], beta_seq[0], gamma_seq[0]
        # randomly initialize Z0, make sure the imaginary part is zero
        Z0 = np.zeros([active_set_z0.sum(), n_coefs*p])*1j \
                + np.random.randn(active_set_z0.sum(), n_coefs*p)*1E-20
        tmp_result = L21solver.solve_stft_regression_tree_group_active_set(
                                M, G_list, G_ind, X, 
                                alpha_star, beta_star, gamma_star,
                                DipoleGroup, DipoleGroupWeight, 
                                Z0, active_set_z0, 
                                active_set_J_ini, n_orient=n_dip_per_pos, 
                                wsize=wsize, tstep=tstep, maxit=maxit, tol=tol,
                                Maxit_J=Maxit_J, Incre_Group_Numb=Incre_Group_Numb,
                                dual_tol=dual_tol, 
                                Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta,
                                Flag_verbose=Flag_verbose)        
        if tmp_result is None:
            # return some results if L2_option == 0 or 1
            #raise Exception("No active dipoles found. alpha is too big.")
            print ("No active dipoles found, returning empty solution")
            return None, np.zeros(n_dipoles, dtype = np.bool), \
                 np.ones(n_step, dtype = np.bool), alpha_star, beta_star, gamma_star, 0, cv_MSE_lasso, 0
            
        Z = tmp_result['Z']
        active_set = tmp_result['active_set']
        active_t_ind = np.ones(n_step, dtype = np.bool)
        # the following part is copied from tf_mixed_norm in v0.11
        if mask is not None:
            active_set_tmp = np.zeros(len(mask), dtype=np.bool)
            active_set_tmp[mask] = active_set
            active_set = active_set_tmp
            del active_set_tmp
        
    # =====================================================================
    delta_star = None # even if L2_option ==0, we will stil return an empty delta_star
    #re-run the regression with a given active set
    if L2_option == 1 or L2_option == 2:
        # if only L2 solution is needed, do some initialization,
        if L2_option == 2: 
            if coef_non_zero_mat is None:
                raise ValueError("if L2_option == 2, coef_non_zero_mat must not be empty!")
            active_set= active_set_z0.copy()
            active_t_ind = np.ones(n_step, dtype = np.bool)
            if Z0_l2 is None:
                # make sure the imaginary part is zero
                Z = np.zeros([active_set_z0.sum(), n_coefs*p])*1j \
                + np.random.randn(active_set_z0.sum(), n_coefs*p)*1E-20
            else:
                Z = Z0_l2
            alpha_star, beta_star, gamma_star = None, None, None
        if L2_option == 1:
            coef_non_zero_mat = np.abs(Z)>0
        if delta_seq is None:
            delta_seq = np.array([1E-12,1E-10,1E-8])
        if len(delta_seq) > 1:
            Z0 = Z.copy()
            Z0 = Z0[:, np.tile(active_t_ind,p*n_freq)]
            delta_star, cv_MSE_L2 = L2solver.select_delta_stft_regression_cv(M,G_list, G_ind, X,
                                                  Z0, active_set, active_t_ind,
                                                  coef_non_zero_mat,
                                                delta_seq,cv_partition_ind,
                                            wsize=wsize, tstep = tstep, 
                                            maxit=maxit, tol = tol,
                                            Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta,
                                            Flag_verbose = Flag_verbose)
        else:
            delta_star = delta_seq[0]
        # L2 optimization
        Z, obj = L2solver.solve_stft_regression_L2_tsparse(M,G_list, G_ind, X, Z, active_set,
                                 active_t_ind, coef_non_zero_mat,
                                 wsize=wsize, tstep = tstep, delta = delta_star,
                                maxit=maxit, tol = tol, 
                                Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta,
                                Flag_verbose = Flag_verbose)
    # =========================================================================
    # reweighting should be done after the debiasing!!!
    # Reapply weights to have correct unit, To Be modifiled
    
    # it seems that in MNE0.11, source_weighting is the inverse of the original source weighting  
    # MNE 0.8 (verified in their 0.81 code "X /= source_weighting[active_set][:, None]")                    
    #Z /= source_weighting[active_set][:, None]
    # MNE 0.11
    Z = _reapply_source_weighting(Z, source_weighting, active_set, n_dip_per_pos)
    Z_full = np.zeros([active_set.sum(),p, n_freq, n_step], dtype = np.complex)
    Z_full[:,:,:,active_t_ind] = np.reshape(Z,[active_set.sum(), p,
                                              n_freq,active_t_ind.sum()])
    Z_full = np.reshape(Z_full, [active_set.sum(),-1])
 
#    do not compute stc_list   
#    tmin = evoked_list[0].times[0]
#    stc_tstep = 1.0 / info['sfreq']
#    stc_list = list()
#    for r in range(n_trials):
#        tmp_stc_data = np.zeros([active_set.sum(),n_times])
#        tmp_Z = np.zeros([active_set.sum(), n_coefs],dtype = np.complex)
#        for i in range(p):
#            tmp_Z += Z_full[:,i*n_coefs:(i+1)*n_coefs]* X[r,i]
#        # if it is a trial by_trial model, add the model for the single trial
#        tmp_stc_data = phiT(tmp_Z)              
#        tmp_stc = _make_sparse_stc(tmp_stc_data, active_set, fwd_list[G_ind[r]], tmin, stc_tstep)
#        stc_list.append(tmp_stc)
#    logger.info('[done]')                               
        
    return Z_full, active_set, active_t_ind, alpha_star, beta_star, gamma_star, delta_star, cv_MSE_lasso, cv_MSE_L2
                     

