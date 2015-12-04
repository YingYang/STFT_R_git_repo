# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:54:14 2014

@author: yingyang
"""

"""
python MNE tutorial example
==============================
Generate simulated evoked data
==============================

# use this to generate non-sparse sources, the dipoles 
# in the same ROI have a same waveform
#generate_stc(src, labels, stc_data, tmin, tstep, value_fun=None)

# create a source label where the values are the distance from the center
#labels = circular_source_labels('sample', 0, 10, 0)

# sources with decaying strength (x will be the distance from the center)
#fun = lambda x: exp(- x / 10)
#stc = generate_stc(fwd, labels, stc_data, tmin, tstep, fun)
"""

#cd Dropbox/MEG_source_loc_proj/Group_Lasso_trial_by_trial/

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

import mne
from mne.viz import plot_evoked, plot_sparse_source_estimates
from mne.simulation import generate_stc
import matplotlib.pyplot as plt
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
mne.set_log_level('warning')

 
import os,sys,inspect
currentdir = "/home/ying/Dropbox/MEG_source_loc_proj/STFT-R/test/"
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from L2_tsparse import solve_stft_regression_L2_tsparse





#============================================================================
# create a spline increasing function
x = np.arange(1,21,1)
n_trials = len(x)
const_a, const_b = 0.1, 0.5
y = (np.log(const_b*x) + const_a + np.random.randn(n_trials))*0.1
internal_knots = np.array([0.5])*n_trials
# linear regression
dm = np.vstack([np.ones(n_trials), np.log(const_b*x) + const_a]).T
coef = (la.lstsq(dm, y))[0]
yhat = np.dot(dm,coef)
plt.plot(x,y)
plt.plot(x,yhat,'g')
plt.show()

#============================================================================
# create the simulated data
#coef are the true coefs, x, yhat are the true signal
Simu_obj = get_simulation(X = dm, coef = coef)

fwd = Simu_obj['fwd']
evoked_list = Simu_obj['evoked_list']
label_ind = Simu_obj['label_ind']
labels = Simu_obj['labels']
stc_data_list = Simu_obj['stc_data_list']
cov = Simu_obj['cov']
snr = Simu_obj['snr']
X = Simu_obj['X']
coef = Simu_obj['coef']

n_channels, n_times = evoked_list[0].data.shape
n_trials = len(evoked_list)
n_channels = 305
M = np.zeros([n_channels, n_times, n_trials])

for r in range(n_trials):
    M[:,:,r] = evoked_list[r].data[0:305,:]

wsize, tstep = 16,4
n_step = int(np.ceil(n_times/float(tstep)))
n_freq = wsize// 2+1
n_coefs = n_step*n_freq
p = 2
n_coefs_all = n_coefs*p
phi = _Phi(wsize, tstep, n_coefs)
phiT = _PhiT(tstep, n_freq, n_step, n_times)

G = fwd['sol']['data'][0:305,:]
n_dipoles = G.shape[1]
DipoleGroup = list([label_ind[0], label_ind[1]])
non_label_ind = np.arange(0,n_dipoles,1)
for i in range(len(label_ind)):
    non_label_ind = np.setdiff1d(non_label_ind, label_ind[i])

for i in range(len(non_label_ind)):
    DipoleGroup.append(np.array([non_label_ind[i]]))
    
true_active_set = np.union1d(label_ind[0], label_ind[1])
DipoleGroupWeight = np.ones(len(DipoleGroup))/np.float(len(DipoleGroup))


# =========== MNE solution ==========
from mne.minimum_norm import make_inverse_operator, apply_inverse
evoked = evoked_list[-1]
inverse_operator = make_inverse_operator(evoked.info, fwd, cov, fixed = True,
                                         loose = None, depth = None)
stc_mne = apply_inverse(evoked, inverse_operator, lambda2=1. / snr**2.,
                         method='MNE')  
                        
thresh = 9E-10
active_set_z0 = np.mean(np.abs(stc_mne.data), axis = 1) > thresh
active_set_z0[true_active_set] = True
#active_set_z0 = np.zeros(n_dipoles,dtype = np.bool) 
print np.nonzero(active_set_z0)
Z0 = phi(stc_mne.data[active_set_z0,:])


# =========== My solution ========= 
import optim_tree_group_lasso_hard_coded_trial_by_trial as pkg
## tuning for alpha and  beta
alpha_seq = np.array([1E-10, 1E-8])
beta_seq = np.array([1E-14, 1E-12])
cv_partition_ind = np.zeros(n_trials, dtype = np.int)
cv_partition_ind[1::2] = 1
best_alpha, best_beta, cv_SSE = pkg.select_alpha_beta_stft_tree_group_cv \
                                         (M,G,X,active_set_z0,
                                         DipoleGroup,DipoleGroupWeight,
                                         alpha_seq, beta_seq,cv_partition_ind,
                                         n_orient=1, wsize=wsize, tstep = tstep, 
                                         maxit=30, tol = 1E-2)
# use the best tuning parameters                                         
Z00 = (np.random.randn(np.sum(active_set_z0), n_coefs*(p+n_trials)) + \
       np.random.randn(np.sum(active_set_z0), n_coefs*(p+n_trials))*1j)*1E-15
Z_hc_all, active_set_hc_all, obj_all = pkg.solve_stft_regression_tree_group_hard_coded_trial_by_trial \
                                (M,G,X,
                                best_alpha,best_beta, 
                                DipoleGroup,DipoleGroupWeight,
                                Z00,active_set_z0,
                                n_orient=1, wsize=wsize, tstep = tstep,
                                maxit=100, tol = 1e-3,lipschitz_constant =None)
print np.nonzero(active_set_hc_all)
print true_active_set
plt.figure()
plt.subplot(4,1,1)
plt.imshow(np.abs(Z_hc_all)) 
plt.subplot(4,1,2)
plt.plot(np.abs(Z_hc_all).T) 
plt.subplot(4,1,3)
plt.plot(phiT(Z_hc_all[:,0:n_coefs]).T)
plt.subplot(4,1,4)
plt.plot(stc0.data.T)
plt.show()
source_ts_hc = np.real(phiT(Z_hc_all[:,0:n_coefs]))
# forgot where _make_sparse_stc comes from
stc_hc = _make_sparse_stc(source_ts_hc, active_set_hc, fwd, tmin = 0, tstep=1)
plot_sparse_source_estimates(fwd['src'], stc_hc, bgcolor=(1, 1, 1),
                             opacity=0.5, high_resolution=True)

# ==================================== the wrapped version ==========

from inverse_regression_stft_trial_by_trial import *
alpha = 1E8
beta = 1E3
Z, active_set,stc_list = get_stc_tree_group_lasso_regression_stft \
                                (evoked_list,X, fwd, cov,
                                labels, 1,
                                stc_mne.data[active_set_z0,:],active_set_z0,
                                alpha, beta,
                                loose=0.2, depth=0.8, maxit=100, tol=1e-2,
                                weights=None, weights_min=None, pca=True,
                                wsize=wsize, tstep=4, window=0.02, 
                                L2_option = 0,gamma = 1E-10,
                                verbose=None)
                               
plot_sparse_source_estimates(fwd['src'], stc_list[-1], bgcolor=(1, 1, 1),
                             opacity=0.5, high_resolution=True)
# plot the estimated Z
plt.figure()
plt.subplot(2,1,1)
plt.imshow(np.abs(Z[:,0:n_coefs]))
plt.colorbar()
plt.title( "abs of Z0 (intercept)")

plt.subplot(2,1,2)
plt.imshow(np.abs(Z[:,1*n_coefs:2*n_coefs]))
plt.colorbar()
plt.title("abs of Z1 (slope)")
plt.show()


true_z_intercept = phi(stc_data)*coef[0]
true_z_slope = phi(stc_data)*coef[1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(true_z_intercept).T)
plt.ylabel("abs of the true intercept")
plt.subplot(2,1,2)
plt.plot(np.abs(true_z_slope).T)
plt.ylabel("abs of the true slope")
plt.show()
true_z = np.hstack([true_z_intercept, true_z_slope])


#=================================
## test MNE
from mne_stft_regression import *
coef, F,sel = mne_stft_regression(evoked_list, inverse_operator,
                                labels, X, snr=1, wsize = wsize, tstep = tstep)
plt.figure()
plt.imshow(np.real(F_array[1,:,:]))
plt.colorbar()
plt.show()  

df1, df2 = p-1, n_trials-p
import scipy
p_array = np.zeros(np.hstack([F_array.shape,2]))
# compute the p-values, for the F test
p_array[:,:,:,0] = 1.0-scipy.stats.f.cdf(np.real(F_array),df1,df2) 
p_array[:,:,:,1] = 1.0-scipy.stats.f.cdf(np.imag(F_array),df1,df2)
# plus FDR comparison?
from mne.stats import fdr_correction
p_array_ravaled = np.ravel(p_array, order = 'C')
reject_array, p_val_corrected = fdr_correction(p_array_ravaled)
reject_array = np.reshape(reject_array, p_array.shape, order = 'C')



i = 40
plt.figure()
plt.subplot(2,3,1)
plt.imshow((reject_array[i,:,:,0]))
plt.title("real fdr results")
plt.subplot(2,3,2)
plt.imshow(np.real(F[i,:,:]))
plt.colorbar()
plt.title("real F")
plt.subplot(2,3,3)
plt.imshow(np.real(coef[i,:,:,0]))
plt.colorbar()
plt.title("real coef")
plt.subplot(2,3,4)
plt.imshow((reject_array[i,:,:,1]))
plt.title("imag fdr results")
plt.subplot(2,3,5)
plt.imshow(np.imag(F[i,:,:]))
plt.colorbar()
plt.title("imag F")
plt.subplot(2,3,6)
plt.imshow(np.imag(coef[i,:,:,0]))
plt.colorbar()
plt.title("imag coef")
plt.show()


# Note: to make the comparison fair, 
# 1) cross-validate and pick the best lambda/snr 
from mne_stft_regression import *
# it takes about 5 min to run 
coef, F,sel = mne_stft_regression(evoked_list, inverse_operator, X,
                          labels = labels,snr=1, wsize = wsize, tstep = tstep)
MSE = get_MSE_mne_stft_regression(evoked_list, fwd, X, coef, labels = labels,
                                  wsize = wsize,
                                  tstep = tstep)
cv_partition_ind = np.zeros(n_trials)
cv_partition_ind[1::2] = 1
snr_tuning_seq = np.array([1.,2.,3.])
best_snr, cv_MSE = select_lambda_tuning_mne_stft_regression_cv(evoked_list,
                                                    inverse_operator,
                                                fwd, X,  cv_partition_ind,
                                                snr_tuning_seq, 
                                                labels = labels, 
                                                wsize=wsize, tstep = tstep)

# 2) the conficence intervals need to be obtained by bootstraping,
# 3) Compare accuracy of the learning curve, and acuracy of the source reconstruction
# 