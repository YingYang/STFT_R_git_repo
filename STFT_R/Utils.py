# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 00:52:33 2014

@author: yingyang
"""

import numpy as np
from mne.inverse_sparse.mxne_optim import tf_lipschitz_constant

#=============================================================================
def get_lipschitz_const(M,G,X,phi,phiT,n_coefs,
                        Flag_trial_by_trial = True, tol = 1e-3):   
    ''' compute the lipschitz constant for the gradient of the smooth part, 
        (sum of squared error)
        using power iteration
        Input: M, n_sensors x n_times x n_trials
               G, n_sensors x n_dipoles
               X, the design matrix
               phi,phiT, the funcitons for stft and istft
               n_coefs, number of coefficients
               maxit, the maximum number of iteration
               Flag_trial_by_trial = True, if true, get the trial by trial coefficients
               tol, the tolerance
        Output, L, the lipschitz constant
        
        This funciton is consistant with tf_lipschitz_constant
    '''
    M0 = M[:,:,0]
    L0 = tf_lipschitz_constant(M0,G,phi,phiT, tol)
    [q,p] = X.shape
    absX = np.abs(X).copy()
    scale = 0
    for i in range(q):
        scale += (absX[i,:].sum() *absX[i,:]).sum()
    if Flag_trial_by_trial:   
        L = np.max([scale *L0,L0]) 
    else:
        L = scale *L0 
    return L
## Old trial by trial version
#    M0 = M[:,:,0]
#    L0 = tf_lipschitz_constant(M0,G,phi,phiT, tol)
#    [q,p] = X.shape
#    absX = np.abs(X).copy()
#    scale = 0
#    for i in range(q):
#        scale += (absX[i,:].sum() *absX[i,:]).sum()
#    L = np.max([scale *L0,L0])    X = dm.copy()
#    return L   