# -*- coding: utf-8 -*-
"""
An extremely simple simulation to test the sparse and non-sparse methods
@author: yingyang
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

import mne
import matplotlib.pyplot as plt
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
mne.set_log_level('warning')

 
import os,sys,inspect
currentdir = "/home/ying/Dropbox/MEG_source_loc_proj/STFT_R/test"
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,"/home/ying/Dropbox/MEG_source_loc_proj")

#from L2_tsparse import solve_stft_regression_L2_tsparse



#import pdb
## ===== use a subset of the true forward matrix =========
# this works for my laptop
#from mne.datasets import sample
#datapath = sample.data_path()
n_channels = 306
n_times = 100
n_dipoles = 600
G = np.random.randn(n_channels, n_dipoles)

G_list = [G, G+1E-1]

## ====== define the groups ========
#n0 = 2
#true_group = np.arange(0,n0)
##DipoleGroup = list([true_group])
#DipoleGroup = list()
#DipoleGroup.append(np.array([0]))
#DipoleGroup.append(np.array([1]))
## note the elements of the list must be arrays
#for i in range(n0,n_dipoles):
#    DipoleGroup.append(np.array([i]).astype(int))

n0 = 6
true_group = np.arange(0,n0)
DipoleGroup = list([true_group])
# note the elements of the list must be arrays
for i in range((n_dipoles-n0)//3):
    DipoleGroup.append(np.array([i*3,i*3+1,i*3+2]).astype(int)+n0)

## ====== create the time series for the true group =====
n_times = 100
wsize, tstep = 16,4
n_step = int(np.ceil(n_times/float(tstep)))
n_freq = wsize// 2+1
n_coefs = n_step*n_freq
phi = _Phi(wsize, tstep, n_coefs)
phiT = _PhiT(tstep, n_freq, n_step, n_times)

n_trials = 20
G_ind = np.zeros(n_trials, dtype = np.int)
G_ind[10::] = 1

X = np.linspace(-0.5, 0.5,n_trials)
M = np.zeros([n_channels, n_times, n_trials])
source_signal = np.zeros([n_dipoles, n_times, n_trials])
m0 = len(true_group)
snr = 1

times = np.arange(0,n_times)
time_course =(np.sin(times*0.1) + np.cos(times*0.9))* 2*np.exp(-(times-20)**2/1E2) 
    #+ 0.1*rd.randn(len(times))
#time_course =  10*np.exp(-(times-50)**2/1E3) 
Z00 = phi(time_course)
#fig = plt.figure()
#fig.add_subplot(2,1,1)
#plt.plot(times, time_course)
#plt.xlabel('time')
#plt.title('time series')
#fig.add_subplot(2,1,2)
#plt.imshow(np.abs(phi(time_course)).reshape([n_freq,n_step]))
#plt.title('absolute value of STFT coeficient')
#plt.show()
#plt.close('all')
Z00 = np.tile(Z00, [m0,1])
Z00 = np.hstack([Z00,Z00*2])

dm = np.vstack([np.ones(len(X)), X]).T
#Z00_true = np.zeros([n_dipoles, n_coefs], dtype = complex)
#Z00_true[true_group,:] = Z00[:,0:n_coefs]
Z00[0,:] *= 0.01;
for i in range(n_trials):
    #tmpZ = Z0* snr*X[i]*(1+1j) + 1.0/snr*(rd.randn(m0,n_coefs) + 1j* rd.randn(m0,n_coefs))
    tmpZ = Z00[:,0:n_coefs]*dm[i,0] + Z00[:,n_coefs:2*n_coefs]*dm[i,1]
    Z = np.zeros([n_dipoles,n_coefs], dtype = complex)
    Z[true_group,:] = tmpZ
    source_signal[:,:,i] = phiT(Z)
    signal = phiT(G_list[G_ind[i]].dot(Z))
    #signal2 = G.dot(source_signal[:,:,i])
    M[:,:,i] = signal + 1/snr* np.random.randn(n_channels, n_times)
    

#fig = plt.figure()
#subplot_array = np.arange(4,20,3)
#for i in range(len(subplot_array)):
#    fig.add_subplot(3,2,i+1)
#    plt.plot(M[:,:,subplot_array[i]].T)  
#plt.show()
#    
    
# test the L2 regression
Z0 = Z00+np.random.randn(6,Z00.shape[1])*10
active_set_z0 = np.zeros(n_dipoles, dtype = np.bool)
active_set_z0[0:6] = True
active_t_ind_z0 = np.ones(n_step, dtype = np.bool)
#coef_non_zero_mat = np.abs(Z00)>=0
coef_non_zero_mat = np.abs(Z00)>=0
lipschitz_constant = 1E8
X = dm
wsize, tstep, delta, maxit, tol = 16,4,10,200, 1E-4

if False:
    from STFT_R.L2_tsparse import solve_stft_regression_L2_tsparse
    tmp1, tmp2 = solve_stft_regression_L2_tsparse(M,G_list, G_ind, X, Z0, 
                                        active_set_z0, active_t_ind_z0,
                                        coef_non_zero_mat,
                                    wsize=16, tstep = 4, delta = delta,
                                    maxit=maxit, tol = 1e-3,lipschitz_constant = None,
                                    Flag_verbose = True)
    plt.figure()
    plt.imshow(np.imag(tmp1))
    plt.figure()
    plt.plot(np.mean(np.real(tmp1), axis = 0))  
    

# test L2 
if True:  
    from STFT_R.L2_tsparse import select_delta_stft_regression_cv
    delta_seq = np.array([ 3, 5, 8])
    cv_partition_ind = np.zeros(n_trials)
    cv_partition_ind[0::2] = 1
    delta_star, cv_MSE = select_delta_stft_regression_cv(M,G_list, G_ind,X,Z0,
                                        active_set_z0, active_t_ind_z0,
                                        coef_non_zero_mat,
                                        delta_seq,cv_partition_ind,
                                        wsize=wsize, tstep = tstep, 
                                        maxit=maxit, tol = 1e-4,
                                        Flag_verbose = False)    
if False:               
    # This is only reasonable when delta is large enough
    from stft_tree_group_lasso.optim_regression_tsparse import solve_stft_regression_L2_tsparse as l2
    tmp1, tmp2 = l2(M,G, X, Z0, 
                                        active_set_z0, active_t_ind_z0,
                                        coef_non_zero_mat,
                                    wsize=16, tstep = 4, delta = delta,
                                    maxit=maxit, tol = 1e-4,lipschitz_constant = None,
                                    Flag_verbose = True, Flag_trial_by_trial = False)
    plt.imshow(np.real(tmp1)) 
    plt.plot(np.mean(np.real(tmp1), axis = 0))                             
                                

# test L21
if True:
    from STFT_R.L21_tsparse import solve_stft_regression_tree_group_tsparse
    alpha = 1E5
    beta = 100
    gamma = 1
    DipoleGroupWeight = np.ones(len(DipoleGroup))/len(DipoleGroup)
    Z_ini = np.random.randn(13,Z00.shape[1])*0.001
    active_set_z_ini = np.zeros(n_dipoles, dtype = np.bool)
    active_set_z_ini[0:13] = True
    active_t_ind_ini = np.ones(n_step, dtype = np.bool)
    n_orient = 1
    Z, active_set_z, active_t_ind_z, obj = \
    solve_stft_regression_tree_group_tsparse(M,G_list, G_ind,X,
                                    alpha,beta, gamma, 
                                    DipoleGroup,DipoleGroupWeight,
                                    Z_ini,active_set_z_ini, active_t_ind_ini,
                                    n_orient=1, wsize= wsize, tstep = tstep,
                                    maxit= 100, tol = tol,lipschitz_constant = None,
                                    Flag_verbose = True)

# for debuging
from STFT_R.Utils import get_lipschitz_const
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
from STFT_R.prox_tree_group_lasso_hard_coded_full_matrix \
                    import (prox_tree_hard_coded_full_matrix,
                            get_tree_norm_hard_coded) 
from STFT_R.sparse_stft import sparse_Phi, sparse_PhiT
from STFT_R.get_gradient import get_gradient0, get_gradient1_L21


from STFT_R.L21_tsparse import (solve_stft_regression_tree_group_tsparse)
from STFT_R.L2_tsparse import get_MSE_stft_regresion_tsparse



# try the active sets algorith
from STFT_R.L21_active_set import (solve_stft_regression_tree_group_active_set, 
                       select_alpha_beta_gamma_stft_tree_group_cv_active_set)
from STFT_R.L21_duality import compute_dual_gap

active_set_J_ini = np.zeros(len(DipoleGroup), dtype = np.bool)
active_set_J_ini[-1] = True
Z_ini = np.random.randn(3, Z00.shape[1])*0.1

active_set_z_ini = np.zeros(n_dipoles, dtype = np.bool)
active_set_z_ini[-3:] = True
alpha, beta, gamma = 1E4, 50, 1
result  = solve_stft_regression_tree_group_active_set(M,G_list, G_ind, X,
                                alpha,beta, gamma, DipoleGroup,DipoleGroupWeight,
                                Z_ini,active_set_z_ini, active_t_ind_ini,
                                active_set_J_ini, 
                                n_orient=1, wsize=wsize, tstep = tstep,
                                maxit=200, tol = 1e-4,
                                Maxit_J = 10, Incre_Group_Numb = 10, dual_tol = 1e-1,
                                Flag_verbose = False)

alpha_seq = np.array([1E6, 1E5, 1E4])
beta_seq = np.array([50])
gamma_seq = np.array([1])
alpha_star, beta_star, gamma_star, cv_MSE = select_alpha_beta_gamma_stft_tree_group_cv_active_set(M,G_list, G_ind,X, 
                                        active_set_J_ini, active_t_ind_ini,
                                         DipoleGroup,DipoleGroupWeight,
                                         alpha_seq, beta_seq, gamma_seq,
                                         cv_partition_ind,
                                         n_orient=1, wsize=wsize, tstep = tstep, 
                                         maxit=200, tol = 1e-4,
                                         Maxit_J = 10, Incre_Group_Numb = 10, 
                                         dual_tol = 1e-1,
                                         Flag_verbose = False)