# -*- coding: utf-8 -*-
"""
Sparse matrix is not used in this implementation. 
If the regression matrix is full, the size is around 8000*250*p, 
for 8000 dipoles and 250 time-frequency components. 
If each complex number is 16 Bytes, then it will be 32M *p in memory. 
A 1G memory can hold p = 32, so p can be no larger than ~10^2.

For larger p, sparse matrix implementation is needed


 
"""

import numpy as np
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
from .prox_tree_group_lasso_hard_coded_full_matrix \
                    import (prox_tree_hard_coded_full_matrix,
                            get_tree_norm_hard_coded) 
from .sparse_stft import sparse_Phi, sparse_PhiT
from .Utils import get_lipschitz_const
from .get_gradient_tsparse import get_gradient0_tsparse, get_gradient1_L21_tsparse

#==============================================================================
def _add_z(z_list, active_set_array):
    """
    A utility function to add z's toghether.
    z_list = [z1, z2, ...], z1, z2 must have the same number of columns
    active_set_array = np.vstack([active_set1, active_set2, ...]),[2,n_dipoles]  boolean
    z1, z2 are the non-zero rows of the full complex coefficient matrices. 
    active_set1 and active_set2 are the active sets.
    
    # test
    n_coef_all = 100; 
    active_set1 = np.zeros(5,dtype = np.bool); 
    active_set1[[0,3]] = True; 
    z1 = np.random.randn(active_set1.sum(),n_coef_all) + np.random.randn(active_set1.sum(),n_coef_all)*1j
    active_set2 = np.zeros(5,dtype = np.bool); 
    active_set2[0:2] = True; 
    z2 = np.random.randn(active_set2.sum(),n_coef_all) + np.random.randn(active_set2.sum(),n_coef_all)*1j
    z_full = np.zeros([len(active_set1),n_coef_all], dtype = np.complex)
    z_full[active_set1] += z1;  z_full[active_set2] += z2;
    z_out, active_set_out= _add_z([z1,z2], np.vstack([active_set1, active_set2]))
    z_full_out = np.zeros([len(active_set1),n_coef_all], dtype = np.complex)
    z_full_out[active_set_out] = z_out
    np.linalg.norm(z_full-z_full_out)
    """
    if active_set_array.shape[0] != len(z_list):
        raise ValueError("_add_z: dimensions do not match.")
    n_coef_all = z_list[0].shape[1]
    active_set_out = np.any(active_set_array,axis = 0)
    if np.any(active_set_out) == 0:
        z_out = np.zeros([0,n_coef_all], dtype = np.complex)
    else:
        z_out = np.zeros([active_set_out.sum(), n_coef_all], dtype = np.complex)
        ind = np.nonzero(active_set_out)[0] 
        for l in range(len(z_list)):
            tmp_indl = np.nonzero(active_set_array[l])[0]
            tmp_ind = [k0 for k0 in range(len(ind)) if ind[k0] in tmp_indl]
            z_out[tmp_ind,:] += z_list[l]
    return z_out, active_set_out
#==============================================================================
def f(Z, active_set_z, M, G_list, G_ind, X, n_coefs, q, p, phiT):
    """
    Compute the smooth objective function, 0.5* sum of squared error
    """
    R_all_sq = 0           
    for r in range(q):
        tmp_coef = np.zeros([active_set_z.sum(),n_coefs], dtype = np.complex)
        for k in range(p):
            tmp_coef += Z[:,k*n_coefs:(k+1)*n_coefs]*X[r,k]
        tmpR = M[:,:,r] - phiT(G_list[G_ind[r]][:,active_set_z].dot(tmp_coef))
        R_all_sq += np.sum(tmpR**2)
    return 0.5*R_all_sq

# =============================================================================== 
def solve_stft_regression_tree_group_tsparse(M,G_list, G_ind,X,
                                alpha,beta, gamma, 
                                DipoleGroup,DipoleGroupWeight,
                                Z_ini, active_set_z_ini, 
                                n_orient=1, wsize=16, tstep = 4,
                                maxit=200, tol = 1e-3,lipschitz_constant = None,
                                Flag_backtrack = True, eta = 1.5, L0 = 1.0,
                                Flag_verbose = False):                             
    """    
    Input:
       M, [n_sensors,n_times,n_trials] array of the sensor data
       G_list,  a list of [n_sensors, n_dipoles] forward gain matrix, for different runs if applicable
       G_ind, [n_trials], index of run number (which G to use)
       X, [n_trials, p]the design matrix, it must include an all 1 colume
       alpha, tuning parameter for the regularization
       beta, the tuning paramter, for the frequency- time basis
       gamma, the penalty on the absolute value of entries in Z
       DipoleGroup, grouping of dipoles,
                    the dipoles in the same ROI are in the same group,
                    the dipoles outside ROIs form one-dipole groups
       DipleGroupWeight, weights of each dipole group (ROI)
       Z_ini, [n_dipoles, n_coefs*p], initial value of Z, the ravel order is [n_dioles, p, n_freqs, n_step]
       active_set_z_ini, [n_dipoles,]  boolean, active_set of dipoles
       n_orient, number of orientations for each dipole
                 note that if n_orient == 3, the DipoleGroup is still on the columns of G, 
                 but grouped in the 3-set way. 
                 The initial values and initial active set of Z should also be correspondingly correct. 
       wsize, number of frequence
       tstep, step in time of the stft
       maxit, maximum number of iteration allowed
       tol, tolerance of the objective function
       lipschitz_constant, the lipschitz constant,
       Flag_backtrack, if True, use backtracking instead of constant stepsize ( the lipschitz constant )
       eta, L0, the shrinking parameters and initial 1/stepsize
       Flag_verbose, if true, print the objective values and difference, else not
    Output:
        Z, the solustion, only rows in the active set, 
             note that it is not guaranteed that all rows from the same group will be in the final solutions,
             if all of the coefficients for that row is zero, it is dropped too. 
        active_set_z,  a boolean vector, active dipoles (rows)
        active_t_ind_z, a boolean vector, active time steps, (time steps)
        obj, the objective function   
    """
    # check the active_set_structure and group structure for n_orient ==3
    if n_orient == 3:
        active_set_mat = active_set_z_ini.copy()
        active_set_mat = active_set_mat.reshape([-1,n_orient])
        any_ind = np.any(active_set_mat,axis =1)
        all_ind = np.all(active_set_mat,axis =1)
        if np.sum(np.abs(any_ind-all_ind)) >0:
            raise ValueError("wrong active set for n_orient = 3")
        # DipoleGroup must also satisfy the structure
        for l in range(len(DipoleGroup)):
            if np.remainder(len(DipoleGroup[l]),n_orient)!=0:
                raise ValueError("wrong group")
            tmp_mat = np.reshape(DipoleGroup[l],[-1,n_orient],order = 'C')
            if np.sum(np.abs(tmp_mat[:,2] - tmp_mat[:,1] - 1)) != 0 \
                    or np.sum(np.abs(tmp_mat[:,1] - tmp_mat[:,0] - 1)) != 0:
                raise ValueError("wrong group")
                
    n_sensors, n_times, q = M.shape
    n_dipoles = G_list[0].shape[1]
    p = X.shape[1]
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq

    # create the sparse and non sparse version of the STFT, iSTFT
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)
    sparse_phi = sparse_Phi(wsize, tstep)
    sparse_phiT = sparse_PhiT(tstep, n_freq, n_step, n_times)
    
    # initialization    
    if lipschitz_constant is None and not Flag_backtrack: 
        lipschitz_constant = 1.1* get_lipschitz_const(M,G_list[0],X,phi,phiT,
                                    Flag_trial_by_trial = False,n_coefs = n_coefs,
                                    tol = 1e-3)
        print "lipschitz_constant = %e" % lipschitz_constant
        
    if Flag_backtrack:
        L = L0   
    else:
        L = lipschitz_constant
    # indices for the active set is only for rows
    Z = Z_ini.copy()
    Y = Z_ini.copy()
    active_set_z = active_set_z_ini.copy()
    active_set_y = active_set_z_ini.copy() 
    
    # my code has sparse time steps. (sparse_phi and sparse_phiT)
    # for consistency, I still kept the active_t_ind variables, 
    # but make sure they are all true, all time steps are used. 
    active_t_ind_full = np.ones(n_step, dtype = np.bool)
    if Z.shape[0] != active_set_z.sum() or Z.shape[1] != n_coefs*p:
        raise ValueError('Z0 shape does not match active sets')
    #==== the main loop =====  
    tau, tau0 = 1.0, 1.0
    obj = np.inf
    old_obj = np.inf
    
    # prepare the M_list from the M
    n_run = len(np.unique(G_ind))
    M_list = list()
    X_list = list()
    for run_id in range(n_run):
        M_list.append(M[:,:,G_ind == run_id])
        X_list.append(X[G_ind == run_id, :])
           
    # greadient part one is fixed   -G^T( \sum_r M(r)) PhiT
    # this should be on all dipoles
    gradient_y0 = get_gradient0_tsparse(M_list, G_list, X_list, p, n_run,
                  n_dipoles, np.ones(n_dipoles, dtype = np.bool), n_times,
                  n_coefs, n_coefs*p, active_t_ind_full,
                  sparse_phi, sparse_phiT)
    # only keep on full matrix
    #gradient_y = np.zeros([n_dipoles, n_coefs*p],dtype =np.complex)
    for i in range(maxit):
        Z0 = Z.copy()
        active_set_z0 = active_set_z.copy()
        # this part can be only on the active set
        # but gradient_y1 should be a full matrix
        gradient_y1 = get_gradient1_L21_tsparse(M_list, G_list, X_list, Y, p, n_run,
                  np.int(active_set_y.sum()), active_set_y, n_times,
                  n_coefs, n_coefs*p, active_t_ind_full,
                  sparse_phi, sparse_phiT)
        gradient_y = gradient_y0 + gradient_y1
  
        # active_set_z0, active rows/dipoles for z0
        # active_t_ind_z0, active columns/time_points for z0
        
        # verify the gradient with two different computations
#        if False:
#            gradient_y20 = np.zeros([n_dipoles, n_coefs*p], dtype = np.complex)
#            gradient_y21 = np.zeros([n_dipoles, n_coefs*p], dtype = np.complex)
#            #GTG_active = G.T.dot(G[:,active_set_y])
#            Y_reshape = np.reshape(Y[:,0:p*n_coefs],[active_set_y.sum(), p,n_coefs])
#            Y_reshape = Y_reshape.swapaxes(1,2)
#            for run_id in range(n_run):
#                GTG_active = G_list[run_id].T.dot(G_list[run_id][:,active_set_y])
#                for k in range(p):
#                    # first term of the gradient
#                    M_sum = np.sum(M_list[run_id]*X_list[run_id][:,k],axis = 2)
#                    # G^T(\sum_r X_k(r) M(r))\Phi
#                    GTM_sumPhi = G_list[run_id].T.dot(sparse_phi(M_sum,active_t_ind_full))
#                    # second term of the gradient
#                    sum_X_coef = np.zeros([active_set_y.sum(), n_coefs], dtype = np.complex)
#                    # the first level is n_coefs, second level is p
#                    for r in range(X_list[run_id].shape[0]):
#                        sum_X_coef += np.sum(Y_reshape*X_list[run_id][r,:],axis = 2)*X_list[run_id][r,k]
#                    gradient_y20[:,k*n_coefs:(k+1)*n_coefs] += -GTM_sumPhi 
#                    gradient_y21 [:,k*n_coefs:(k+1)*n_coefs] += sparse_phi(sparse_phiT(GTG_active.dot(sum_X_coef),
#                                                          active_t_ind_full),active_t_ind_full)
#            #print (np.linalg.norm(gradient_y0-gradient_y20)/np.linalg.norm(gradient_y20)) 
#            #print (np.linalg.norm(gradient_y1-gradient_y21)/np.linalg.norm(gradient_y21)) 
#            gradient_y2 = gradient_y20+gradient_y21                                              
#            print "diff gradient=%e" % (np.linalg.norm(gradient_y-gradient_y2)/np.linalg.norm(gradient_y2)) 
#            #plt.plot(np.real(gradient_y2).ravel(),np.real(gradient_y).ravel(), '.' )
#            gradient_y = gradient_y2.copy()
        
        tmp_Y_L_gradient_Y = -gradient_y/L
        tmp_Y_L_gradient_Y[active_set_y,:] += Y
        ## ==== ISTA step, get the proximal operator ====
        ## Input must be a full matrix, so the active set is set to full
        Z, active_set_z = prox_tree_hard_coded_full_matrix(tmp_Y_L_gradient_Y, n_coefs, p, 
                                 alpha/L, beta/L, gamma/L,
                                 DipoleGroup, DipoleGroupWeight, n_orient)
        # check if it is zero solution                       
        if not np.any(active_set_z):
            print "active_set = 0"
            return 0,0,0                
        objz = f( Z, active_set_z, M, G_list, G_ind, X, n_coefs, q, p, phiT)
        # compute Z-Y
        diff_z, active_set_diff_z = _add_z([Z,-Y],  np.vstack([active_set_z, active_set_y]))
        
        if Flag_backtrack:
            objy = f( Y, active_set_y, M, G_list, G_ind, X, n_coefs, q, p, phiT)
            # compute the criterion for back track: f(z)-f(y) -grad_y.dot(z-y) -0.5*(z-y)**2
            # note gradient_y is a full matrix
            # note diff_z is complex
            # http://www.seas.ucla.edu/~vandenbe/236C/lectures/fgrad.pdf 7-17+ FISTA paper Beck
            diff_bt = objz-objy- np.sum( np.real(gradient_y[active_set_diff_z])* np.real(diff_z))\
                      - np.sum(np.imag(gradient_y[active_set_diff_z])* np.imag(diff_z)) \
                      -0.5*L*np.sum( np.abs(diff_z)**2)
            while diff_bt > 0:
                L = L*eta
                tmp_Y_L_gradient_Y = -gradient_y/L
                tmp_Y_L_gradient_Y[active_set_y,:] += Y
                Z, active_set_z = prox_tree_hard_coded_full_matrix(tmp_Y_L_gradient_Y, n_coefs, p, 
                                         alpha/L, beta/L, gamma/L,
                                         DipoleGroup, DipoleGroupWeight, n_orient)
                if not np.any(active_set_z):
                    print "active_set = 0"
                    return 0,0,0                        
                objz = f( Z, active_set_z, M, G_list, G_ind, X, n_coefs, q, p, phiT)
                # Z-Y
                diff_z, active_set_diff_z = _add_z([Z,-Y],  np.vstack([active_set_z, active_set_y]))
                # the criterion for back track: 
                diff_bt = objz-objy- np.sum( np.real(gradient_y[active_set_diff_z])* np.real(diff_z))\
                      - np.sum(np.imag(gradient_y[active_set_diff_z])* np.imag(diff_z)) \
                      -0.5*L*np.sum( np.abs(diff_z)**2)                
        
        ## ==== FISTA step, update the variables =====
        tau0 = tau;
        tau = 0.5*(1+ np.sqrt(4*tau**2+1))
        
        if np.any(active_set_diff_z) == 0:
            print "active_set = 0"
            return 0,0,0
        else:
            # y <- z + (tau0-1)/tau (z-z0) =(tau+tau0-1)/tau z - (tao0-1)/tau z0
            Y, active_set_y = _add_z( [(tau+tau0-1)/tau*Z, -(tau0-1)/tau*Z0],
                                       np.vstack([active_set_z, active_set_z0]))
            
            # commented after verification
            #ind_y = np.nonzero(active_set_y)[0]
            #Z_larger = np.zeros([active_set_y.sum(), n_coefs*p], dtype = np.complex)
            #tmp_intersection = np.all(np.vstack([active_set_y, active_set_z]),axis=0)
            #tmp_ind_inter = np.nonzero(tmp_intersection)[0]
            #tmp_ind = [k0 for k0 in range(len(ind_y)) if ind_y[k0] in tmp_ind_inter]
            #Z_larger[tmp_ind,:] = Z
            # expand Z0 
            #Z0_larger = np.zeros([active_set_y.sum(), n_coefs*p], dtype = np.complex)
            #tmp_intersection0 = np.all(np.vstack([active_set_y, active_set_z0]),axis=0)
            #tmp_ind_inter0 = np.nonzero(tmp_intersection0)[0]
            #tmp_ind0 = [k0 for k0 in range(len(ind_y)) if ind_y[k0] in tmp_ind_inter0]        
            #Z0_larger[tmp_ind0,:] = Z0
            ## update y
            #diff = Z_larger-Z0_larger
            #Y1 = Z_larger + (tau0 - 1.0)/tau *diff 
            #print "Y1-Y=%f" %np.linalg.norm(Y1-Y)
            
        ## ===== compute objective function, check stopping criteria ====
        old_obj = obj
        full_Z = np.zeros([n_dipoles, n_coefs*p], dtype = np.complex)
        full_Z[active_set_z, :] = Z

        # residuals in each trial 
        # if we use the trial by trial model, also update 
        # the gradient of the trial-by-trial coefficients 
        # debug, can be commented soon.                         
        #R_all_sq = 0           
        #for r in range(q):
        #    tmp_coef = np.zeros([active_set_z.sum(),n_coefs], dtype = np.complex)
        #    for k in range(p):
        #        tmp_coef += Z[:,k*n_coefs:(k+1)*n_coefs]*X[r,k]
        #    tmpR = M[:,:,r] - phiT(G_list[G_ind[r]][:,active_set_z].dot(tmp_coef))
        #    R_all_sq += np.sum(tmpR**2)    
        #print "diff in f(z) = %f" %(objz - 0.5* R_all_sq.sum() )
        obj = objz \
            + get_tree_norm_hard_coded(full_Z,n_coefs, p, 
              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient)
        diff_obj = old_obj-obj
        relative_diff = np.sum(np.abs(diff_z))/np.sum(np.abs(Z))
        if Flag_verbose: 
            print "\n iteration %d" % i
            print "diff_obj = %e" % (diff_obj/obj)
            print "obj = %e" %obj
            print "diff = %e" %relative_diff
        stop = ( relative_diff < tol  and np.abs(diff_obj/obj) < tol)        
        if stop:
            print "convergence reached!"
            break    
    Z = Y.copy() 
    active_set_z = active_set_y.copy()
    return Z, active_set_z, obj
