# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:10:34 2014

@author: yingyang
"""

import numpy as np
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
from .sparse_stft import sparse_Phi, sparse_PhiT
#import pdb
from .get_gradient import get_gradient0, get_gradient1_L21

# To be modeified!!
# =============================================================================== 
def compute_dual_gap(M, G_list, G_ind, X, Z, active_set, 
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = 16, tstep = 4):
    """ 
    Compute the duality gap, and check the feasibility of the dual function.
    Input:
        M, G_list, G_ind, X, Z, active_set, all the full primal.
        alpha, beta, gamma, penalty paramters
        n_orient, number of orientations
    Output:
        dict(feasibility_dist = feasibility_dist,
                gradient = gradient,
                feasibility_dist_DipoleGroup = feasibility_dist_DipoleGroup )       
        feasibility_dist, is the major criterion, if it is small enougth, then accept the results
        gradient, the gradient
    """
    # initialization, only update when seeing greater values
    if gamma == 0:
        raise ValueError( "the third level must be penalized!!")
    
    n_dipoles = G_list[0].shape[1]
    n_sensors, n_times, q= M.shape
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)
    sparse_phi = sparse_Phi(wsize, tstep)
    sparse_phiT = sparse_PhiT(tstep, n_freq, n_step, n_times)    
    n_trials, p = X.shape
    active_t_ind_full = np.ones(n_step, dtype = np.bool)
            
    
    n_run = len(np.unique(G_ind))
    M_list = list()
    X_list = list()
    for run_id in range(n_run):
        M_list.append(M[:,:,G_ind == run_id])
        X_list.append(X[G_ind == run_id, :])
    
    
    
    # compute the gradient to check feasibility
    gradient_y0 = get_gradient0(M_list, G_list, X_list, p, n_run,
                  n_dipoles, np.ones(n_dipoles, dtype = np.bool), n_times,
                  n_coefs, n_coefs*p, active_t_ind_full,
                  sparse_phi, sparse_phiT)
    gradient_y1 = get_gradient1_L21(M_list, G_list, X_list, Z, p, n_run,
              np.int(active_set.sum()), active_set, n_times,
              n_coefs, n_coefs*p, active_t_ind_full,
              sparse_phi, sparse_phiT)
    gradient = gradient_y0 + gradient_y1
  
    # sanity check
    # for each dipole, get the maximum abs value among the real and imag parts
    max_grad = np.max(np.vstack([np.max(np.abs(np.real(gradient)), axis = 1), 
                   np.max(np.abs(np.imag(gradient)),axis = 1)]), axis = 0)
                   
    alpha_weight = np.zeros(n_dipoles)
    for i in range(len(DipoleGroup)):
        alpha_weight[DipoleGroup[i]] = alpha*DipoleGroupWeight[i]

    # if max_grad is greater than alpha+beta+gamma, 
    # then the dual variable can not be feasible
 
    #if np.any(max_grad > alpha_weight+ beta+gamma):
    #     feasibility_dist = np.inf
    #else:
    #    
    # feasibility check. b = A^T u + \sum_g D_g^T v_g
    # where g is the non zero groups 
    b = gradient.copy()
    active_set_ind = np.nonzero(active_set)[0]
    Z_full = np.zeros([n_dipoles, Z.shape[1]],dtype = np.complex)
    Z_full[active_set,:] = Z
    
    # add g in the alpha level, dipole groups 
    # a bool vector showing whether the group is in the active set
    DipoleGroup_active = np.zeros(len(DipoleGroup), dtype = np.bool)
    for i in range(len(DipoleGroup)):
        if np.intersect1d(DipoleGroup[i],active_set_ind).size >0:
            DipoleGroup_active[i] = True
            # l2 norm of the group
            l2_norm_alpha = np.sqrt(np.sum( ( np.abs(Z_full[DipoleGroup[i],:]) )**2) )
            # add the sum of the gradient
            if l2_norm_alpha == 0:
                raise ValueError("all zero in an active group!")
            b[DipoleGroup[i],:]+= Z_full[DipoleGroup[i],:]/l2_norm_alpha * alpha_weight[i]
    
    # add g in the beta level, same dipole, same stft coef
    if n_orient == 1:
        Z_reshape = np.reshape(Z, [active_set.sum(), p, -1])
        # active_set.sum() x  n_coefs
        l2_norm_beta = np.sqrt(np.sum( (np.abs(Z_reshape))**2, axis = 1))
        # active_set.sum() x  n_coefs * p
        l2_norm_beta_large = np.tile(l2_norm_beta,[ 1, p])                
       
    else: # n_orient == 3
        Z_reshape = np.reshape(Z, [active_set.sum()//3,3, p, -1])
        l2_norm_beta = np.sqrt( np.sum(   np.sum( ( np.abs(Z_reshape) )**2, axis = 2) , axis = 1 ) )
        l2_norm_beta_large = np.reshape(  np.tile(l2_norm_beta,[ 1, 3*p]) , Z.shape)

    tmp_add_to_b = np.zeros([active_set.sum(), Z.shape[1]], dtype = np.complex)
    nonzero_beta = l2_norm_beta_large > 0
    tmp_add_to_b[nonzero_beta] = Z[nonzero_beta]/l2_norm_beta_large[nonzero_beta]*beta
    b[active_set,:] += tmp_add_to_b
    
    # add g in the gamma level, each element of the matrix
    if n_orient == 1:
        l2_norm_gamma = np.abs(Z)   
    else: # n_orient == 3:
        Z_reshape = np.reshape(Z, [active_set.sum()//3,3,-1])
        l2_norm_gamma = np.sqrt ( np.sum( ( np.abs(Z_reshape) )**2, axis = 1 ) )
        l2_norm_gamma = np.reshape( np.tile(l2_norm_gamma, [1,3* Z.shape[1]]), Z.shape)
        
    nonzero_gamma = l2_norm_gamma > 0
    tmp_add_to_b = np.zeros([active_set.sum(), Z.shape[1]], dtype = np.complex)
    tmp_add_to_b[nonzero_gamma] = Z[nonzero_gamma]/l2_norm_gamma[nonzero_gamma]*gamma
    b[active_set,:] += tmp_add_to_b 
    if np.any(np.isnan(b)):
            raise ValueError("nan found in b!")
    # use coordinate descent to solve the feasibility problem
    nonzero_Z = np.abs(Z) >0 
    feasibility_result = get_feasibility(b, active_set, DipoleGroup, 
                                 alpha_weight, beta, gamma,
                                 n_coefs, p,  nonzero_Z, 
                                 DipoleGroup_active) 
    feasibility_dist =  feasibility_result['feasibility_dist']
    # the dist in each dipole, squared
    feasibility_dist_in_alpha_level = feasibility_result['feasibility_dist_in_alpha_level'] 
    feasibility_dist_DipoleGroup = np.zeros(len(DipoleGroup))
    for i in range(len(DipoleGroup)):
        feasibility_dist_DipoleGroup[i] = \
              np.sqrt( np.sum(feasibility_dist_in_alpha_level[DipoleGroup[i]]**2))                        
    
         
    return dict(feasibility_dist = feasibility_dist,
                gradient = gradient,
                feasibility_dist_DipoleGroup = feasibility_dist_DipoleGroup)   

            
def get_feasibility( b, active_set, DipoleGroup, alpha_weight, beta, gamma,
                    n_coefs, p,  nonzero_Z, DipoleGroup_active, 
                    n_orient = 1,  maxit = 100, tol = 1e-5):
    '''
    Input:
        b, [n_dipoles, n_coefs*p]
        active_set, [n_dipoles,], boolean
        DipoleGroup
        alpha_weight
        beta,gamma
        n_coefs, p
        nonzero_Z, [active_set.sum(), n_coefs*p], boolean
        DipoleGroup_active, [n_group,], boolean, is the group active
        maxit, tol   
        
    Output: 
        the distance to the feasible set, sqrt((Au+\Dg vg)**2)
    '''
    n_dipoles = len(active_set)
    nonzero_Z_full = np.zeros([n_dipoles, n_coefs*p], dtype = np.bool)
    nonzero_Z_full[active_set,:] = nonzero_Z
    # v_g can be represented with [Z_full.shape, 3] matrix, with some entries fixed to zero
    vg = np.zeros([n_dipoles, n_coefs*p, 3], dtype = np.complex)
    #v_g[nonzero_Z_full] = 0
    # initialization 
    # residual
    r = b+ np.sum(vg, axis = 2)
    obj = np.sum(np.abs(r)**2)
    old_obj = obj
    diff = np.Inf
    # gradient is  r|g
    for i in range(maxit):
        # block coordinate descent, update at gamma, beta, alpha levels
        # ===================gamma levels, vg[:,:,2] 
        candidate = -(b + np.sum(vg[:,:, np.array([0,1])], axis = 2))
        if n_orient == 1:
            L2_gamma = np.abs(candidate)
            L2_gamma[L2_gamma < gamma] = gamma
            shrink = gamma/L2_gamma
            # the non zero indices to exclude
            tmp_non_zero_ind = nonzero_Z_full.copy()
        else : # n_orient == 3
            tmp = np.reshape(candidate, [n_dipoles//3, 3, n_coefs*p])
            L2_gamma = np.sqrt(np.sum(np.abs(tmp)**2, axis = 1))
            L2_gamma[L2_gamma < gamma] = gamma
            shrink = gamma/L2_gamma
            shrink = np.tile(shrink, [1,3])
            shrink = np.reshape(shrink, [n_dipoles, n_coefs*p])
            # the non zero indices to exclude
            tmp_non_zero_ind = nonzero_Z_full.copy()
            tmp_non_zero_ind = np.any(np.reshape(tmp_non_zero_ind, 
                                                 [n_dipoles//3, 3, n_coefs*p]), axis = 1)
            tmp_non_zero_ind = np.reshape(np.tile(tmp_non_zero_ind, [1,3]), [n_dipoles, n_coefs*p])                                     
        new_vg = candidate*shrink
        # set the non-zero groups to zero
        new_vg[tmp_non_zero_ind] = 0.0
        vg[:,:,2] = new_vg       
        #=================== beta levels, vg[:,:,1]
        candidate = -(b + np.sum(vg[:,:,np.array([0,2])], axis = 2))
        if n_orient == 1:
            L2_beta = np.sqrt(np.sum(np.reshape(np.abs(candidate)**2,[n_dipoles, p, -1]),axis = 1))
            L2_beta[L2_beta < beta] = beta
            shrink = beta/L2_beta
            shrink_mat = np.tile(shrink, [1, p])
            # the non zero indices
            tmp_non_zero_ind = nonzero_Z_full.copy()
            tmp_non_zero_ind = np.any(np.reshape(tmp_non_zero_ind, 
                                                 [n_dipoles, p, n_coefs]), axis = 1)
            tmp_non_zero_ind = np.tile(tmp_non_zero_ind,[1,p])                                    
        else: # n_orient = 3
            L2_beta = np.sqrt(np.sum(np.sum(np.reshape(np.abs(candidate)**2,
                                                [n_dipoles//3,3, p, -1]),axis = 2),
                                                axis = 1))
            L2_beta[L2_beta < beta] = beta
            shrink = beta/L2_beta
            shrink_mat = np.tile(shrink, [1, p*3])
            shrink_mat = np.reshape(shrink_mat, [n_dipoles, n_coefs*p])
             # the non zero indices
            tmp_non_zero_ind = nonzero_Z_full.copy()
            tmp_non_zero_ind = np.any(np.any(np.reshape(tmp_non_zero_ind, 
                                                 [n_dipoles//3, 3, p, n_coefs]), axis = 2),
                                                 axis = 1)
            tmp_non_zero_ind = np.reshape(np.tile(tmp_non_zero_ind,[1,3*p]),
                                          [n_dipoles, n_coefs*p] )
        new_vg = candidate*shrink_mat
        new_vg[tmp_non_zero_ind] = 0.0
        vg[:,:,1] = new_vg
        #============ alpha_levels, vg[:,:,0]
        candidate = -(b + np.sum(vg[:,:, np.array([1,2]) ], axis = 2))
        L2_alpha = np.zeros(n_dipoles)
        for j in range(len(DipoleGroup)):
           tmp = np.sqrt(np.sum(np.abs(candidate[DipoleGroup[j],:])**2))
           L2_alpha[DipoleGroup[j]] = tmp* np.ones(len(DipoleGroup[j]))
        
        L2_alpha[L2_alpha < alpha_weight] = alpha_weight[L2_alpha < alpha_weight]
        # some of the L2_alpha are zero, to avoid NaN, set those L2_alpha as np.inf
        L2_alpha[L2_alpha == 0] = np.inf
        shrink = alpha_weight/L2_alpha
        new_vg = (candidate.T*shrink).T
        for j in range(len(DipoleGroup)):
            if DipoleGroup_active[j]:
                new_vg[DipoleGroup[j],:] = 0.0
                
        vg[:,:,0] = new_vg
        # update the residual
        r = b+ np.sum(vg, axis = 2)
        old_obj = obj
        obj= np.sum(np.abs(r)**2)
        diff = old_obj - obj
        # debug
        if np.any(np.isnan(vg)):
            pdb.set_trace()
            print vg
            print b
            raise ValueError("nan found!")
        print "diff = %f obj = %f" %(diff, obj)
        if diff < tol:
            break
    feasibility_dist = np.sqrt(obj)
    feasibility_dist_in_alpha_level = np.sqrt(np.sum(np.abs(r)**2, axis = 1))
    result = dict(feasibility_dist = feasibility_dist,
                  feasibility_dist_in_alpha_level = feasibility_dist_in_alpha_level)
    return result
        
        
    
    
                        