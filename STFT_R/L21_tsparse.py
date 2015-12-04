# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:10:34 2014

@author: yingyang
"""

import numpy as np
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
from .prox_tree_group_lasso_hard_coded_full_matrix \
                    import (prox_tree_hard_coded_full_matrix,
                            get_tree_norm_hard_coded) 
from .sparse_stft import sparse_Phi, sparse_PhiT
from .Utils import get_lipschitz_const
from .get_gradient import get_gradient0, get_gradient1_L21

# =============================================================================== 
def solve_stft_regression_tree_group_tsparse(M,G_list, G_ind,X,
                                alpha,beta, gamma, 
                                DipoleGroup,DipoleGroupWeight,
                                Z_ini,active_set_z_ini, active_t_ind_ini,
                                n_orient=1, wsize=16, tstep = 4,
                                maxit=200, tol = 1e-3,lipschitz_constant = None,
                                Flag_verbose = False):                             
    """
    note: I made a mistake about the active set, now after the fixation,
         active_set_z_ini will not be used, it will alwayes be set to all true instead. 
         Z, Z0, Y will still be sparse,
             but the input to the proximal operator will always be full.        
         therefore, Z_ini has to be [n_active_dipoles,]
    
    Input:
       M, [n_sensors,n_times,n_trials] array of the sensor data
       G_list,  a list of [n_sensors, n_dipoles] forward gain matrix
       G_ind, [n_trials], index of run number (which G to use)
       X, [n_trials, p]the design matrix, it must include an all 1 colume
       alpha, tuning parameter for the regularization
       beta, the tuning paramter of balance between single frequency- time basis and dipoles
       gamma, the penalty on the absolute value of entries in Z
       DipoleGroup, grouping of dipoles,
                    the dipoles in the same ROI are in the same group,
                    the dipoles outside ROIs form one-dipole groups
       DipleGroupWeight, weights of each dipole group (ROI)
       active_set_z_ini, [n_dipoles,]  boolean, active_set of dipoles
       active_t_ind_ini, [nstep,] boolean, active set of time indices
       Z_ini, [n_dipoles, n_coefs*(p+q) or p], initial value of Z
            each row, pq * n_freq * n_step_active
       n_orient, number of orientations for each dipole
                 note that if n_orient == 3, the DipoleGroup is still on the columns of G, 
                 but grouped in the 3-set way. 
                 The initial values and initial active set of Z should also be correspondingly correct. 
       wsize, number of frequence
       tstep, step in time of the stft
       maxit, maximum number of iteration allowed
       tol, tolerance of the objective function
       lipschitz_constant, the lipschitz constant
       Flag_verbose, if true, print the objective values and difference, else not
    Output:
        Z, the solustion, only rows in the active set, 
             note that it is not guaranteed that all rows from the same group will be in the final solutions,
             if all of the coefficients for that row is zero, it is dropped too. 
        active_set_z,  a boolean vector, active dipoles (rows)
        active_t_ind_z, a boolean vector, active time steps, (time steps)
        obj, the objective function   
    """
    n_sensors, n_times, q = M.shape
    n_dipoles = G_list[0].shape[1]
    p = X.shape[1]
    n_step = int(np.ceil(n_times/float(tstep)))
    n_freq = wsize// 2+1
    n_coefs = n_step*n_freq

    # create the sparse and non sparse version
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)
    sparse_phi = sparse_Phi(wsize, tstep)
    sparse_phiT = sparse_PhiT(tstep, n_freq, n_step, n_times)
    # check whether active_t_ind_ini is  all true, can be removed. 
    if active_t_ind_ini.sum()*n_freq*p!= Z_ini.shape[1] \
         or n_dipoles != len(active_set_z_ini):
        print active_t_ind_ini.sum()*n_freq*p, Z_ini.shape[1],\
             n_dipoles, len(active_set_z_ini)
        raise ValueError("wrong number of dipoles or coefs")
    if lipschitz_constant is None: 
        lipschitz_constant = 1.1* get_lipschitz_const(M,G_list[0],X,phi,phiT,
                                    Flag_trial_by_trial = False,n_coefs = n_coefs,
                                    tol = 1e-3)
    print "lipschitz_constant = %e" % lipschitz_constant
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
    # indices for the active set is only for rows
    Z = Z_ini.copy()
    Y = Z_ini.copy()
    active_set_z = active_set_z_ini.copy()
    active_set_y = active_set_z_ini.copy() 
    # the active_t_ind are alwayse fully true,
    # this the result of my wrong code, for consistency, I still kept the active_t_ind variables
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
    gradient_y0 = get_gradient0(M_list, G_list, X_list, p, n_run,
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
        gradient_y1 = get_gradient1_L21(M_list, G_list, X_list, Y, p, n_run,
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
                                                        
        full_Y=-gradient_y/ lipschitz_constant
        full_Y[active_set_y,:] += Y

        ## ==== ISTA step, get the proximal operator ====
        #  flatten the array into a vector, row major, so the grouping should be row major too
        ## the proximal operator must be hard coded to save time of creating groups,
        ## "it must also take the sparse structure into consideration"
        ##  correction, this must be a full matrix instead, so the active set is set to full
        Z, active_set_z = prox_tree_hard_coded_full_matrix(full_Y,
                                 n_coefs, p, alpha/lipschitz_constant,
                                 beta/lipschitz_constant, gamma/lipschitz_constant,
                                 DipoleGroup, DipoleGroupWeight, n_orient)                     
        if not np.any(active_set_z):
            print "active_set = 0"
            return 0,0,0,0
        ## ==== FISTA step, update the variables =====
        tau0 = tau;
        tau = 0.5*(1+ np.sqrt(4*tau**2+1))
        # new active set y
        active_set_y = np.any(np.vstack([active_set_z, active_set_z0]),axis = 0)
        if np.any(active_set_y) == 0 or not np.any(active_set_z0):
            print "active_set = 0"
            return 0,0,0,0
        else:
            # make Z and Z0 the same size, get back to [n_active_dipoles, n_coefs_all]
            ind_y = np.nonzero(active_set_y)[0]
            Z_larger = np.zeros([active_set_y.sum(), n_coefs*p], dtype = np.complex)
            tmp_intersection = np.all(np.vstack([active_set_y, active_set_z]),axis=0)
            tmp_ind_inter = np.nonzero(tmp_intersection)[0]
            tmp_ind = [k0 for k0 in range(len(ind_y)) if ind_y[k0] in tmp_ind_inter]
            Z_larger[tmp_ind,:] = Z
            # expand Z0 
            Z0_larger = np.zeros([active_set_y.sum(), n_coefs*p], dtype = np.complex)
            tmp_intersection0 = np.all(np.vstack([active_set_y, active_set_z0]),axis=0)
            tmp_ind_inter0 = np.nonzero(tmp_intersection0)[0]
            tmp_ind0 = [k0 for k0 in range(len(ind_y)) if ind_y[k0] in tmp_ind_inter0]        
            Z0_larger[tmp_ind0,:] = Z0
            # update y
            diff = Z_larger-Z0_larger
            Y = Z_larger + (tau0 - 1.0)/tau *diff   
        ## ===== compute objective function, check stopping criteria ====
        old_obj = obj
        full_Z = np.zeros([n_dipoles, n_coefs*p], dtype = np.complex)
        full_Z[active_set_z, :] = Z
        
        # residuals in each trial 
        # if we use the trial by trial model, also update 
        # the gradient of the trial-by-trial coefficients                         
        R_all_sq = 0           
        for r in range(q):
            tmp_coef = np.zeros([active_set_z.sum(),n_coefs], dtype = np.complex)
            for k in range(p):
                tmp_coef += Z[:,k*n_coefs:(k+1)*n_coefs]*X[r,k]
            tmpR = M[:,:,r] - phiT(G_list[G_ind[r]][:,active_set_z].dot(tmp_coef))
            R_all_sq += np.sum(tmpR**2)
        obj = 0.5* R_all_sq.sum() \
            + get_tree_norm_hard_coded(full_Z,n_coefs, p, 
              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient)
        diff_obj = old_obj-obj
        if Flag_verbose: 
            print "iteration %d" % i
            print "diff_obj = %e" % diff_obj
            print "obj = %e" %obj
            print "diff = %e" % (np.abs(diff).sum()/np.sum(abs(Y)))
        stop = (np.abs(diff).sum()/np.sum(abs(Y)) < tol  and np.abs(diff_obj/obj) < tol)        
        if stop:
            print "convergence reached!"
            break    
    Z = Y.copy() 
    active_set_z = active_set_y.copy()
    active_t_ind_z = active_t_ind_full 
    return Z, active_set_z, active_t_ind_z, obj
