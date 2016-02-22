# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:10:34 2014

@author: yingyang
"""

import numpy as np
from copy import deepcopy
from .L21_duality import compute_dual_gap
from .L21_tsparse import (solve_stft_regression_tree_group_tsparse)
from .L2_tsparse import get_MSE_stft_regresion_tsparse

## To Be Verified
# ==========================================================================
def solve_stft_regression_tree_group_active_set(M,G_list, G_ind, X,
                                alpha,beta, gamma, DipoleGroup,DipoleGroupWeight,
                                Z_ini,active_set_z_ini, 
                                active_set_J_ini, 
                                n_orient=1, wsize= 16, tstep = 4,
                                maxit=200, tol = 1e-3,
                                Maxit_J = 10, Incre_Group_Numb = 50, dual_tol = 1e-2,
                                Flag_verbose = False,
                                Flag_backtrack = True, L0 = 1.0, eta = 1.1):
    '''
    Use the active set strategy to compute the tree_group_lasso_solution                                    
    Input:
        M,
        G_list,
        G_ind
        X,
        alpha, beta, gamma
        DipoleGroup,
        DipoleGroupWeight,
        Z_ini,
        active_set_z_ini,

        active_set_J_ini, the initial active set for the optimization, 
                 [n_groups,] bool
        n_orient,
        wsize,
        tstep,
        maxit, tol
        Maxit_J, the maxinum number of optimizations
        Incre_Group_Numb, number of groups to add to the active set
                 for the next optimization
        dual_tol, tolerance of feasibility dist
    Output:
        Z, active_set, active_set_J
    '''
    active_set_J = deepcopy(active_set_J_ini)
    n_dipoles = G_list[0].shape[1]
    Z = Z_ini.copy()
    active_set = active_set_z_ini.copy()
    n_run = len(G_list)
    for i in range(Maxit_J):
        tmp_DipoleGroup =  [DipoleGroup[j] for j in range(len(DipoleGroup))
                         if active_set_J[j] ]                 
        tmp_DipoleGroupWeight = DipoleGroupWeight[active_set_J]
        tmp_active_dipoles = np.zeros(n_dipoles, dtype = np.bool)
        for j in range(len(tmp_DipoleGroup)):
            tmp_active_dipoles[tmp_DipoleGroup[j]] = True
            
        tmp_active_dipoles_ind = np.nonzero(tmp_active_dipoles)[0]
        # make the groups correct, i.e. match with the reduced G
        tmp_DipoleGroup1 = deepcopy(tmp_DipoleGroup)
        for j in range(len(tmp_DipoleGroup)):
            tmp = tmp_DipoleGroup1[j]
            for k in range(len(tmp)):
                 tmp_ind = np.where(tmp_active_dipoles_ind==tmp_DipoleGroup[j][k])
                 tmp[k] = tmp_ind[0][0]
            tmp_DipoleGroup1[j] = tmp
        # run the optimization
        # note, active_set is always [n_dipoles,]
        tmpG_list = list()
        for run_id in range(n_run):
            tmpG_list.append(G_list[run_id][:, tmp_active_dipoles])
        
        # tmp_active_dipoles is alwayse a super set of active_set, due to the greedy increment of number of groups
        # so I did not need to check if Z.shape[0] does not match tmp_active_set.sum()
        tmp_active_set = active_set[tmp_active_dipoles]
        Z_full = np.zeros([n_dipoles, Z.shape[1]], dtype = np.complex)
        Z_full[active_set,:] = Z
        Z = Z_full[tmp_active_dipoles,:]
        Z = Z[tmp_active_set,:]
        Z, tmp_active_set, obj = solve_stft_regression_tree_group_tsparse\
                                (M,tmpG_list, G_ind,X,alpha,beta,gamma,
                                tmp_DipoleGroup1,tmp_DipoleGroupWeight,
                                Z, tmp_active_set, 
                                n_orient=n_orient, wsize=wsize, tstep = tstep,
                                maxit=maxit, tol = tol,  Flag_verbose = Flag_verbose,
                                Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta)  
        if isinstance(Z, np.int):
            print "zero solution"
            return None
        active_set = np.zeros(n_dipoles, dtype=np.bool) 
        active_set[tmp_active_dipoles] = tmp_active_set
        dual_result = compute_dual_gap(M, G_list, G_ind, X, Z, active_set, 
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = wsize, tstep = tstep)

        feasibility_dist = dual_result['feasibility_dist']
        gradient = dual_result['gradient']
        feasibility_dist_DipoleGroup = dual_result['feasibility_dist_DipoleGroup']
        
        non_active_group = np.nonzero(1-active_set_J)[0]  
        non_active_group_feas_dist = feasibility_dist_DipoleGroup[non_active_group]
        # total violation in the non-active groups
        non_active_group_feas_dist_total = np.sqrt(np.sum(non_active_group_feas_dist**2))
        
        relative_feas_dist = feasibility_dist/np.linalg.norm(gradient)
        relative_feas_dist_non_active = non_active_group_feas_dist_total/np.linalg.norm(gradient)
        print " feasibility_dist/ norm(gradient) = %f" %  relative_feas_dist
        print " feasibility_dist_non_active =%f " % non_active_group_feas_dist_total
        print " feasibility_dist_non_active/ norm(gradient) = %f" %  relative_feas_dist_non_active
        if  relative_feas_dist_non_active  < dual_tol:
            print "feasibility dist in non-active group small enough"
            break
        else: 
            ## method 1, add the group that has the largest abs of gradient
            #max_gradient_each_group = np.zeros(len(DipoleGroup))
            #for j in range(len(DipoleGroup)):
            #    max_gradient_each_group[j] = np.max(np.abs(gradient[DipoleGroup[j],:]))
            ## find the Incre_Group_Numb that have the largest gradient
            #non_active_group_gradient = max_gradient_each_group[non_active_group]
            ## descending sorting, by multiplying -1
            #sorted_non_active_group = np.argsort(-1.0*non_active_group_gradient)
            ## add some to the active set
            #n_groups_to_add = min(len(sorted_non_active_group),Incre_Group_Numb)
            #active_set_J[sorted_non_active_group[0:n_groups_to_add]] = True
            ## debug
            #print "Method 1"
            #print sorted_non_active_group[0:n_groups_to_add]
            
            
            # method 2 add the group with the largest violaiton of KKT cond. 
            # i.e. the feasibility distance
            sorted_non_active_group = np.argsort(-1.0*non_active_group_feas_dist)
            n_groups_to_add = min(len(sorted_non_active_group),Incre_Group_Numb)
            active_set_J[non_active_group[sorted_non_active_group[0:n_groups_to_add]]]  \
                                       = True
            ## debug
            #print "Method 2"
            #print non_active_group[sorted_non_active_group[0:n_groups_to_add]]
            print "# active groups = %d"  % active_set_J.sum()
                        
            
    result = dict(Z= Z, active_set = active_set, 
                      obj = obj, active_set_J = active_set_J, 
                      feasibility_dist = feasibility_dist)
    return result
    
#===========================================================================            
def select_alpha_beta_gamma_stft_tree_group_cv_active_set(M,G_list, G_ind,X, 
                                        active_set_J_ini, 
                                         DipoleGroup,DipoleGroupWeight,
                                         alpha_seq, beta_seq, gamma_seq,
                                         cv_partition_ind,
                                         n_orient=1, wsize= 16, tstep = 4, 
                                         maxit=200, tol = 1e-3,
                                         Maxit_J = 10, Incre_Group_Numb = 50, 
                                         dual_tol = 1e-2,
                                         Flag_verbose = False,
                                         Flag_backtrack = True, L0 = 1.0, eta = 1.1): 

    ''' Find the best L1 regularization parameter gamma by cross validation
       Note that here, in training, the trial by trial paramter is estimated, 
       but in testing, only the regression coefficients were used. 
       Remember to set the weights to the ROIs to be 0, then alpha is 
       a easy variable to tune. 

    Input:
        active_set_J_ini, the initial active set for the optimization, 
                 [n_groups,] bool
       
    Output:
       alpha_star, beta_star, the best tuning pramters 
       cv_MSE, the cv MSE for all combintations of alpha and betas
           MSE = mean squared error across trials
    '''
    n_sensors, n_times, n_trials = M.shape
    n_dipoles = G_list[0].shape[1]    
    p = X.shape[1] 
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq

    n_fold = len(np.unique(cv_partition_ind))
    n_alpha = len(alpha_seq)
    n_beta = len(beta_seq)
    n_gamma = len(gamma_seq)
    cv_MSE = np.zeros([n_alpha, n_beta, n_gamma, n_fold], dtype = np.float)
    cv_MSE.fill(np.Inf)
                             
    for k in range(n_fold):  
        # partition
        test_trials = np.nonzero(cv_partition_ind == k)[0]
        train_trials = np.nonzero(cv_partition_ind != k)[0]
        
        n_coefs_all_active = n_coefs*p
           
        Mtrain = M[:,:,train_trials]
        Xtrain = X[train_trials,:]
        Mtest = M[:,:,test_trials]
        Xtest = X[test_trials,:] 
        G_ind_train = G_ind[train_trials]
        G_ind_test = G_ind[test_trials]
        
        for l in range(n_gamma):                                 
            for j in range(n_beta):
                # initial value
                active_set_J = deepcopy(active_set_J_ini)
                # initial values and active sets
                tmp_DipoleGroup =  [DipoleGroup[j0] for j0 in range(len(DipoleGroup))
                             if active_set_J[j0] ]                 
                tmp_active_dipoles = np.zeros(n_dipoles, dtype = np.bool)
                for i in range(len(tmp_DipoleGroup)):
                    tmp_active_dipoles[tmp_DipoleGroup[j]] = True
                active_set_z_ini = tmp_active_dipoles 
                # make sure the imaginary part is zero!
                Z_ini = (np.random.randn(active_set_z_ini.sum(), n_coefs_all_active) \
                        + np.zeros([active_set_z_ini.sum(), n_coefs_all_active])*1j)*1E-20
                active_set_z_ini_tmp = active_set_z_ini.copy()
               
                for i in range(n_alpha):
                    tmp_alpha, tmp_beta, tmp_gamma = alpha_seq[i], beta_seq[j], gamma_seq[l]
                    # training  
                    result = solve_stft_regression_tree_group_active_set(Mtrain,G_list,G_ind_train,
                                                                         Xtrain,
                                    tmp_alpha,tmp_beta,tmp_gamma,
                                    DipoleGroup,DipoleGroupWeight,
                                    Z_ini,active_set_z_ini_tmp,
                                    active_set_J, 
                                    n_orient=n_orient, wsize=wsize, tstep = tstep,
                                    maxit=maxit, tol = tol,
                                    Maxit_J = Maxit_J, Incre_Group_Numb = Incre_Group_Numb, 
                                    dual_tol = dual_tol,
                                    Flag_verbose = Flag_verbose,
                                    Flag_backtrack = Flag_backtrack, L0 = L0, eta = eta)
                    if result is None:
                        # if this happened, no need to test the following beta
                        print "zero solution"
                        continue                
                    Z, active_set_z = result['Z'], result['active_set']  
                    
                    active_set_J = result['active_set_J']
                    # update the initial value
                    # because beta is ascending, the previous
                    # result is a good intial value
                    Z_ini = Z.copy()
                    active_set_z_ini_tmp = active_set_z.copy()
                   
                    # only take the regression coefficients out
                    Z_star = Z[:,0:p*n_freq*n_step]
                    # testing
                    tmp_val, _,_,_ =get_MSE_stft_regresion_tsparse(Mtest,G_list,G_ind_test,Xtest,
                                    Z_star, active_set_z, np.ones(n_step, dtype = bool),
                                    wsize=wsize, tstep = tstep)
                    cv_MSE[i,j,l,k] = tmp_val   
    # row for alpha, and columns for beta  
    cv_MSE = np.mean(cv_MSE, axis = 3)
    best_ravel_ind = np.argmin(np.ravel(cv_MSE, order = 'C'))
    best_i, best_j, best_l = np.unravel_index(best_ravel_ind, cv_MSE.shape, order = 'C')
    alpha_star = alpha_seq[best_i]
    beta_star = beta_seq[best_j]
    gamma_star = gamma_seq[best_l]
    return alpha_star, beta_star, gamma_star, cv_MSE
    