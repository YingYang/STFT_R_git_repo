# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:10:34 2014
L2 regression

@author: yingyang
"""

import numpy as np
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
from .sparse_stft import sparse_Phi, sparse_PhiT
from .Utils import get_lipschitz_const
from .get_gradient import get_gradient0, get_gradient1

# allow for different G's for different runs

# ================================================
def solve_stft_regression_L2_tsparse(M,G_list, G_ind, X, Z0, 
                                    active_set_z0, active_t_ind_z0,
                                    coef_non_zero_mat,
                                wsize=16, tstep = 4, delta = 0,
                                maxit=200, tol = 1e-3,lipschitz_constant = None,
                                Flag_verbose = False):                             
    """
    Use the accelerated gradient descent (exactly FISTA without non-smooth penalty)
        to find the solution given an active set
        min 1/2||R||_F^2 + delta ||Z||_F^2
    Input:
       M, [n_channels, n_times, n_trials] array of the sensor data
       G_list, a list of [n_channels, n_dipoles] forward gain matrix
       G_ind, [n_trial], marks the index of G for this run
       X, [n_trials, p],the design matrix, it must include an all 1 colume
       Z0, [n_active_dipoles, p*n_freq * n_active_step]
       acitve_set_z0, [n_dipoles,] a boolean array, indicating the active set of dipoles
       active_t_ind_z0, [n_step, ] a boolean array, indicating the active set of time points,
           the union of all frequencies, columns of X, and dipoles
       coef_non_zero_mat,[n_active_dipoles,n_coef*pq]  boolean matrix,
           since some active_set_z and active_t_ind_z is a super set of the active set,
           but we assume that it is the UNION of each coefficient of X and all trials
       wsize, window size of the STFT
       tstep, length of the time step
       delta, the regularization parameter
       maxit, maximum number of iteration allowed
       tol, tolerance of the objective function
       lipschitz_constant, the lipschitz constant
       
       No Flag trial by trial is allowed in this version
    """
    n_sensors, n_times, q= M.shape
    n_dipoles = G_list[0].shape[1]
    p = X.shape[1]
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq

    #coef_non_zero_mat_full = np.tile(coef_non_zero_mat,[1,pq])
    coef_non_zero_mat_full = coef_non_zero_mat
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)
    sparse_phi = sparse_Phi(wsize, tstep)
    sparse_phiT = sparse_PhiT(tstep, n_freq, n_step, n_times)
    
    if (active_t_ind_z0.sum()*n_freq*p != Z0.shape[1] or n_dipoles!= len(active_set_z0)):
        print active_t_ind_z0.sum()*n_freq*p, Z0.shape[1],n_dipoles,len(active_set_z0)
        raise ValueError("wrong number of dipoles or coefs")
    # use the first G in G_list to estimate the lipschiz contant
    if lipschitz_constant is None: 
        lipschitz_constant = 1.1*( get_lipschitz_const(M,G_list[0],X,phi,phiT,n_coefs,
                         tol = 1e-3,Flag_trial_by_trial = False) +2.0*delta)                                             
    #initialization
    active_t_ind_z = active_t_ind_z0.copy()
    active_set_z = active_set_z0.copy()
    Z = Z0.copy()
    Z[coef_non_zero_mat_full==0] = 0
    Y = Z0.copy()    
    Y[coef_non_zero_mat_full==0] = 0
    n_active_dipole = active_set_z.sum()
    # number of coeficients s
    n_coefs_z = n_freq*active_t_ind_z.sum()
    n_coefs_all_active = p*n_coefs_z
    #==== the main loop =====  
    tau, tau0 = 1.0, 1.0
    obj = np.inf
    old_obj = np.inf
    obj0 = np.inf
    
    # prepare the M_list from the M
    n_run = len(np.unique(G_ind))
    M_list = list()
    X_list = list()
    for run_id in range(n_run):
        M_list.append(M[:,:,G_ind == run_id])
        X_list.append(X[G_ind == run_id, :])
           
    # greadient part one is fixed   -G^T( \sum_r M(r)) PhiT
    gradient_y0 = get_gradient0(M_list, G_list, X_list, p, n_run,
                  n_active_dipole, active_set_z, n_times,
                  n_coefs_z, n_coefs_all_active, 
                  active_t_ind_z,
                  sparse_phi, sparse_phiT)
        
    #  iterations, we only need to compute the second part of gradient 
        #  +G^T G(\sum_r X_k^(r) \sum_k Z_k  X_k^r) PhiT Phi
    for i in range(maxit):
        Z0 = Z.copy()
        gradient_y1 = get_gradient1(M_list, G_list, X_list, Y, p, n_run,
                  n_active_dipole, active_set_z, n_times,
                  n_coefs_z, n_coefs_all_active, 
                  active_t_ind_z,
                  sparse_phi, sparse_phiT)
        gradient_y = gradient_y0 + gradient_y1
        # compare the gradient(tested)
#        if False: 
#            gradient_y2 = np.zeros([n_active_dipole, n_coefs_all_active], dtype =np.complex)
#            R_all_sq = np.zeros([n_sensors, n_times])
#            for r in range(q):
#                # tmp_coef = y(0) + \sum_k y(k)* X(r,k)
#                tmp_coef = np.zeros([n_active_dipole,n_coefs_z], dtype = np.complex)
#                for k in range(p):
#                    tmp_coef += Y[:,k*n_coefs_z:(k+1)*n_coefs_z]*X[r,k]
#                # current residual for this trial            
#                tmpR = np.real(M[:,:,r] - G_list[G_ind[r]][:,active_set_z].dot(sparse_phiT(tmp_coef, active_t_ind_z)))           
#                R_all_sq += np.abs(tmpR)**2
#                tmpA = G_list[G_ind[r]][:,active_set_z].T.dot(sparse_phi(tmpR, active_t_ind_z))
#                for k in range(p):
#                    gradient_y2[:,k*n_coefs_z:(k+1)*n_coefs_z] += - X[r,k]*tmpA 
#            print "obj = %e" %(0.5* R_all_sq.sum() + delta*np.sum(np.abs(Y)**2))
#            #plt.plot(np.real(gradient_y2.ravel()), np.real(gradient_y.ravel()), '.')
#            #plt.plot(np.real(gradient_y2.ravel()), np.real(gradient_y2.ravel()), 'r')
#            print np.linalg.norm(gradient_y2-gradient_y)/np.linalg.norm(gradient_y2)
#            #gradient_y = gradient_y2.copy()
        # the L2 pentalty
        gradient_y += 2*delta*Y
        # compute the variable  y- 1/L gradient y
        gradient_y[coef_non_zero_mat_full==0] =0
        Y -= gradient_y/lipschitz_constant
        ## ==== ISTA step, get the proximal operator ====
        Z = Y.copy()
        ## ==== FISTA step, update the variables =====
        tau0 = tau;
        tau = 0.5*(1+ np.sqrt(4*tau**2+1))
        diff = Z-Z0
        Y = Z + (tau0 - 1.0)/tau* diff
        ## ===== compute objective function, check stopping criteria ====
        old_obj = obj
        obj = np.linalg.norm(gradient_y)
        diff_obj = old_obj-obj
        if i ==0:
            obj0 = obj
            
        if Flag_verbose:
            print "iteration %d" % i
            print "diff_obj = %e" % diff_obj
            print "norm gradient = %e" %obj
            print "diff = %e" %(np.abs(diff).sum()/np.sum(abs(Y)))
            #print "diff_obj/obj0 = %e" % np.abs(diff_obj/obj0)
        #stop = (np.abs(diff).sum()/np.sum(abs(Y)) < tol  and np.abs(diff_obj/obj0) < tol)
        stop = (np.abs(diff).sum()/np.sum(abs(Y)) < tol) 
        if stop:
            print "convergence reached!"
            break    
          
    Z = Y.copy() 
    Z[coef_non_zero_mat_full ==0] =0
    return Z, obj
        
# =============================================================================        
def get_MSE_stft_regresion_tsparse(M,G_list, G_ind,X,
                                Z0, active_set_z0, active_t_ind_z0,
                                wsize=16, tstep = 4):                             
    """
    Evaluate the MSE, given a sparse Z, the active_t_ind,
    Input:
       M, [n_channels, n_times, n_trials] array of the sensor data
       G_list, a list of [n_channels, n_dipoles] forward gain matrix
       G_ind, [n_trial], marks the index of G for this run
       X, [n_trials, p],the design matrix, it must include an all 1 colume
       Z0, [n_active_dipoles, p*n_freq * n_active_step]
       acitve_set_z0, [n_dipoles,] a boolean array, indicating the active set of dipoles
       active_t_ind_z0, [n_step, ] a boolean array, indicating the active set of time points,
           the union of all frequencies, columns of X, and dipoles
       wsize, window size of the STFT
       tstep, length of the time step
       
    Output:
       MSE, residual, stc_data,dipole_active_set
    """

    n_sensors, n_times, q = M.shape
    n_dipoles = G_list[0].shape[1]
    p = X.shape[1]
    
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    
    sparse_phiT = sparse_PhiT(tstep, n_freq, n_step, n_times)

    Z = Z0.copy()
    active_set_z = active_set_z0.copy()
    active_t_ind_z = active_t_ind_z0.copy()
    n_active_dipole = active_set_z.sum()
    
    if (active_t_ind_z.sum()*n_freq*p != Z.shape[1]  \
                   or n_dipoles != len(active_set_z)):
        raise ValueError("wrong number of dipoles or coefs")

    R_all_sq = np.zeros([n_sensors, n_times], dtype = np.float)
    stc_data = np.zeros([n_active_dipole,n_times, q], dtype = np.float)
    residual = np.zeros([n_sensors, n_times, q], dtype = np.float)
    n_coefs_z = active_t_ind_z.sum()*n_freq
    ##==== compute the gradient =====         
    for r in range(q):
        tmp_coef = np.zeros([n_active_dipole, n_coefs_z], dtype = np.complex)
        for k in range(p):
            tmp_coef += Z[:,k*n_coefs_z:(k+1)*n_coefs_z]*X[r,k]
            
        tmp_stc_data = np.real(sparse_phiT(tmp_coef, active_t_ind_z))
        tmpR = M[:,:,r]-G_list[G_ind[r]][:,active_set_z].dot(tmp_stc_data)
        residual[:,:,r] = tmpR
        stc_data[:,:,r] = tmp_stc_data            
        R_all_sq += np.abs(tmpR)**2
        
    MSE = 0.5* R_all_sq.sum()/np.float(q)
    return MSE, residual, stc_data, active_set_z
        
        
#=====================================================                
# select the best L2 regularization parameter
def select_delta_stft_regression_cv(M,G_list, G_ind,X,Z00,
                                    active_set_z0, active_t_ind_z0,
                                    coef_non_zero_mat,
                                    delta_seq,cv_partition_ind,
                                    wsize=16, tstep = 4, 
                                    maxit=200, tol = 1e-3,
                                    Flag_verbose = False): 
    ''' Find the best L2 regularization parameter delta by cross validation
        Note that here, in training, the trial by trial paramter is estimated, 
        but in testing, only the regression coefficients were used. 

    Input:
        M, [n_channels, n_times, n_trials] array of the sensor data
        G_list, a list of [n_channels, n_dipoles] forward gain matrix
        G_ind, [n_trial], marks the index of G for this run
        X, [n_trials, p],the design matrix, it must include an all 1 colume
        Z00, [n_active_dipoles, p*n_freq * n_active_step]
        e.g 
            # initial value
                Z00 = (np.random.randn(n_true_dipoles, n_coefs_all_active) \
                + np.random.randn(n_true_dipoles, n_coefs_all_active)*1j)*1E-15  
                n_coefs_all_active = active_t_ind_z0.sum()*n_freq*pq
                
        acitve_set_z0, [n_dipoles,] a boolean array, indicating the active set of dipoles
        active_t_ind_z0, [n_step, ] a boolean array, indicating the active set of time points,
            the union of all frequencies, columns of X, and dipoles
        coef_non_zero_mat,[n_active_dipoles,n_coefs*pq]  boolean matrix,
            since some active_set_z and active_t_ind_z is a super set of the active set,
            but we assume that it is the UNION of each coefficient of X and all trials
        delta_seq, np.array, a sequence of delta to be tested
        cv_parition_ind, an integer array of which cross-validation group
                    each trial is in
        wsize, window size of the STFT
        tstep, length of the time step
        maxit, maximum number of iteration
        tol, tolerance,
    Output:
       delta_star, the best delta  
       cv_MSE, the cv MSE for all elements in delta_seq
    '''
    n_sensors, n_times, n_trials = M.shape
    n_dipoles = G_list[0].shape[1]
    if len(active_set_z0) != n_dipoles:
        raise ValueError("the number of dipoles does not match") 
    p = X.shape[1]
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)
    n_fold = len(np.unique(cv_partition_ind))
    n_delta = len(delta_seq)
    # n_true_dipoles = np.sum(active_set_z0)
    lipschitz_constant0 = get_lipschitz_const(M,G_list[0],X,phi,phiT,n_coefs,
                          tol = 1e-3,Flag_trial_by_trial = False)
    cv_MSE = np.zeros([n_fold, n_delta])
    
    for j in range(n_fold):
        # partition
        test_trials = np.nonzero(cv_partition_ind == j)[0]
        train_trials = np.nonzero(cv_partition_ind != j)[0]
        Z0 = Z00
        tmp_coef_non_zero_mat =  coef_non_zero_mat
     
        Mtrain = M[:,:,train_trials]
        Xtrain = X[train_trials,:]
        Mtest = M[:,:,test_trials]
        Xtest = X[test_trials,:] 
        G_ind_train = G_ind[train_trials]
        G_ind_test = G_ind[test_trials]
        for i in range(n_delta):
            tmp_delta = delta_seq[i]
            # lipschitz constant
            L = (lipschitz_constant0+ 2*tmp_delta)*1.1
            # training  
            Z, _ = solve_stft_regression_L2_tsparse (Mtrain,G_list,
                                                     G_ind_train, Xtrain, Z0,
                        active_set_z0, active_t_ind_z0, tmp_coef_non_zero_mat,
                        wsize=wsize, tstep =tstep,delta = tmp_delta,
                        maxit=maxit,tol = tol,lipschitz_constant =L,
                        Flag_verbose = Flag_verbose)
            # only take the regression coefficients out
            Z_star = Z[:,0:p*active_t_ind_z0.sum()*n_freq]
            Z0 = Z.copy()
            # testing
            tmp_val, _,_,_ = get_MSE_stft_regresion_tsparse(Mtest,G_list,
                                           G_ind_test,Xtest,
                                 Z_star, active_set_z0, active_t_ind_z0,
                                 wsize=wsize, tstep = tstep)
            cv_MSE[j,i] =  tmp_val
            # debug
            #import matplotlib.pyplot as plt
            #plt.figure()
            #plt.plot(np.real(Z_star).T)
            #plt.title(tmp_delta)
            
    cv_MSE = np.mean(cv_MSE, axis = 0)
    best_ind = np.argmin(cv_MSE)
    delta_star = delta_seq[best_ind]
    return delta_star, cv_MSE
          