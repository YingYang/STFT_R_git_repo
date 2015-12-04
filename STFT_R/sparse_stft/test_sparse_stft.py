# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:01:33 2014
Test Sparse STFT
@author: yingyang
"""


import numpy as np
from mne.time_frequency.stft import stft, istft
from mne.time_frequency import morlet
import matplotlib.pyplot as plt
from sparse_stft import sparse_stft, sparse_istft, sparse_Phi, sparse_PhiT


sfreq = 100.0
Ws = morlet(sfreq, [3, 10], n_cycles=[1, 1.5])
stc_data = np.zeros([2,100])
stc_data[0][:len(Ws[0])] = np.real(Ws[0])
stc_data[1][:len(Ws[1])] = np.real(Ws[1])
x = stc_data
plt.plot(x.T)
plt.show()
n_times = x.shape[1]

wsize = 16
tstep = 4
n_freq = wsize // 2 + 1
n_step = n_step = int(ceil(n_times / float(tstep)))
sparse_phi = sparse_Phi(wsize, tstep)
sparse_phiT = sparse_PhiT(tstep, n_freq, n_step, n_times)

# =============== test stft====================================================
X0= stft(x, wsize = wsize, tstep = tstep )
X0_norm = np.abs(X0[0,:,:,])
active_t_ind = np.any(X0_norm>1E-3, axis=0) 
X1_sparse = sparse_stft(x,wsize, tstep, active_t_ind)
X1 = np.zeros(X0.shape, dtype =np.complex)
X1[:,:, active_t_ind] = X1_sparse

X1_sparse_vec = sparse_phi(x,active_t_ind)
X1_sparse_vec_mat = np.reshape(X1_sparse_vec,[2,n_freq,-1])
print np.linalg.norm(X1_sparse - X1_sparse_vec_mat)

# order, 3d array, the outer part is the first, the innter part is the last
# C order:
# ‘C’ means to read / write the elements using C-like index order,
# with the last axis index changing fastest,
# back to the first axis index changing slowest.
# all sliced arrays are views of the original array,
# i.e. if the original array changes, the sliced array changes
# and vise versa

# X1.ndim = X1_sparse.ndim


i = 0
plt.figure()
plt.subplot(2,2,1)
plt.imshow(np.real(X0[i,:,:]))
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(np.imag(X0[i,:,:]))
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(np.real(X1[i,:,:]))
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(np.imag(X1[i,:,:]))
plt.colorbar()
plt.show()

# test if phi is still linear, it does not seems neumerically linear?
A = np.random.randn(2,2)
X1_sparse_vec1 = sparse_phi(A.dot(x),active_t_ind)
X1_sparse_vec2 = A.dot(sparse_phi(x,active_t_ind))
print np.linalg.norm(X1_sparse_vec1-X1_sparse_vec2) \
    /np.linalg.norm(X1_sparse_vec1)

#===========  Note, a potential caveate of the mne's sparse representation=====
# even if the stft coefs Z are sparse in time, after phiH and phi, it would be
# sparse, the non-active set will have very small values near zero. 
# this may be why my algorithms are very slow, 
# and the L2 version of it does not work well

# ================ test istft==================================================

x0 = istft(X0, tstep = tstep)
x1 = sparse_istft(X1_sparse,tstep, active_t_ind)
x1_phi = sparse_phiT(X1_sparse_vec,active_t_ind)
plt.figure()
plt.plot(x0.T,'r')
plt.plot(x1.T,'g')
plt.plot(x.T,'b')
plt.plot(x1_phi.T,'m')
plt.show()

x11 = sparse_phiT(A.dot(X1_sparse_vec), active_t_ind)
x12 = A.dot(sparse_phiT(X1_sparse_vec, active_t_ind))
print np.linalg.norm(x11-x12) \
    /np.linalg.norm(x12)


# ============ caveate of using the union temporal active set =================
# all active sources have the same temporal active sets, 
# some of them may be actually zero, 
# there has to be a good thresholding after the L1 selection, 
# or Take the raw output Z, avoid doing phiH, phi again.





