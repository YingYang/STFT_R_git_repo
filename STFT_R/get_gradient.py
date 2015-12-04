# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:05:56 2015

@author: ying
"""

import numpy as np

def get_gradient0(M_list, G_list, X_list, p, n_run,
                  n_active_dipole, active_set_z, n_times,
                  n_coefs_z, n_coefs_all_active, 
                  active_t_ind_z,
                  sparse_phi, sparse_phiT):
    # for the smooth sum of squared error, compute the first part of the gradient
    gradient_y0 = np.zeros([n_active_dipole, n_coefs_all_active], dtype =np.complex)
    for k in range(p):
        # sum over all runs
        tmp = np.zeros([n_active_dipole, n_times])
        for run_id in range(n_run):
            # sum over all trials in the run
            tmp_sum = np.sum(M_list[run_id]* X_list[run_id][:,k],axis = 2)
            tmp += -G_list[run_id][:,active_set_z].T.dot(tmp_sum)  
        # we only need one time of STFT    
        gradient_y0[:, k*n_coefs_z:(k+1)*n_coefs_z] = sparse_phi(tmp, active_t_ind_z)
    return gradient_y0
        
#======================= this only applies to the L2 constrainted version======
def get_gradient1(M_list, G_list, X_list, Y, p, n_run,
                  n_active_dipole, active_set_z, n_times,
                  n_coefs_z, n_coefs_all_active, 
                  active_t_ind_z,
                  sparse_phi, sparse_phiT):
    # get the gradient for the second part                 
    gradient_y1 =   np.zeros([n_active_dipole, n_coefs_all_active], dtype =np.complex)                   
    for k in range(p):
        tmp = np.zeros([n_active_dipole, n_coefs_z], dtype = np.complex)
        for run_id in range(n_run):
            tmp0 = np.zeros([n_active_dipole, n_coefs_z], dtype = np.complex)
            # sum k0  Z_k0\sum_r   X_k^r X_k0^r
            for k0 in range(p):
                #  Z_k \sum_r  X_k^r X_k0^r
                tmp0 += Y[:,k0*n_coefs_z:(k0+1)*n_coefs_z] * np.sum(X_list[run_id][:,k0]* X_list[run_id][:,k])
            tmp+= np.dot(G_list[run_id][:, active_set_z].T, 
                               np.dot(G_list[run_id][:,active_set_z],tmp0))
        gradient_y1[:,k*n_coefs_z:(k+1)*n_coefs_z] += sparse_phi(sparse_phiT(tmp, active_t_ind_z), active_t_ind_z)
    return gradient_y1
    
#==================== this only applie to the L21 algorithms ================ 
# in this case, we need G^T G[:, active_set]
def get_gradient1_L21(M_list, G_list, X_list, Y, p, n_run,
                  n_active_dipole, active_set_z, n_times,
                  n_coefs_z, n_coefs_all_active, 
                  active_t_ind_z,
                  sparse_phi, sparse_phiT):
    # get the gradient for the second part 
    n_dipoles = len(active_set_z)
    gradient_y1 =   np.zeros([n_dipoles, n_coefs_all_active], dtype =np.complex)                   
    for k in range(p):
        tmp = np.zeros([n_dipoles, n_coefs_z], dtype = np.complex)
        for run_id in range(n_run):
            tmp0 = np.zeros([n_active_dipole, n_coefs_z], dtype = np.complex)
            # sum k0  Z_k0\sum_r   X_k^r X_k0^r
            for k0 in range(p):
                #  Z_k \sum_r  X_k^r X_k0^r
                tmp0 += Y[:,k0*n_coefs_z:(k0+1)*n_coefs_z] * np.sum(X_list[run_id][:,k0]* X_list[run_id][:,k])
            tmp+= np.dot(G_list[run_id].T, np.dot(G_list[run_id][:,active_set_z],tmp0))
        gradient_y1[:,k*n_coefs_z:(k+1)*n_coefs_z] += sparse_phi(sparse_phiT(tmp, active_t_ind_z), active_t_ind_z)
    return gradient_y1