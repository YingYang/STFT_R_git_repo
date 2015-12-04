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
compare the regression results between my method and MNE, using the wrapped version
"""

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
# laptop
#os.chdir('/home/yingyang/Dropbox/MEG_source_loc_proj/stft_tree_group_lasso/')
# desktop
os.chdir('/home/ying/Dropbox/MEG_source_loc_proj/stft_tree_group_lasso/')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+"/Simulation") 
sys.path.insert(0,parentdir + "/Spline_Regression") 
sys.path.insert(0,parentdir + "/MNE_stft")
sys.path.insert(0,currentdir)
from mne.inverse_sparse.mxne_optim import *
from Simulation_real_scale import *
from get_b_splines import *


#============================================================================
# create a spline increasing function
x = np.arange(1,21,1)
n_trials = len(x)
const_a, const_b = 0.1, 0.5
y = (np.log(const_b*x) + const_a + np.random.randn(n_trials)*0.1)
internal_knots = np.array([0.5])*n_trials
B_basis = get_b_spline_dm(x,internal_knots,deg = 3) 

# linear regression
dm = np.hstack([np.ones([n_trials,1]),B_basis])
coef = (la.lstsq(dm, y))[0]
yhat = np.dot(dm,coef)
#plt.plot(x,y)
#plt.plot(x,yhat,'g')
#plt.show()

#============================================================================
# create the simulated data
#coef are the true coefs, x, yhat are the true signal
Simu_obj = get_simulation(n_trials = n_trials, X = dm, coef = coef, snr = 2.0, 
                          wsize_phi = 16, tstep_phi = 4)

fwd = Simu_obj['fwd']
evoked_list = Simu_obj['evoked_list']
label_ind = Simu_obj['label_ind']
labels = Simu_obj['labels']
true_stc_data_list = Simu_obj['stc_data_list']
cov = Simu_obj['cov']
snr = Simu_obj['snr']
X = Simu_obj['X']
coef = Simu_obj['coef']
Z_true = Simu_obj['Z_true']


n_channels, n_times = evoked_list[0].data.shape
n_trials = len(evoked_list)

wsize, tstep = 16,4
n_step = int(np.ceil(n_times/float(tstep)))
n_freq = wsize// 2+1
n_coefs = n_step*n_freq
p = 2
n_coefs_all = n_coefs*p
phi = _Phi(wsize, tstep, n_coefs)
phiT = _PhiT(tstep, n_freq, n_step, n_times)

G = fwd['sol']['data']
n_dipoles = G.shape[1]
DipoleGroup = list([label_ind[0], label_ind[1]])
non_label_ind = np.arange(0,n_dipoles,1)
for i in range(len(label_ind)):
    non_label_ind = np.setdiff1d(non_label_ind, label_ind[i])

for i in range(len(non_label_ind)):
    DipoleGroup.append(np.array([non_label_ind[i]]))
    
true_active_set = np.union1d(label_ind[0], label_ind[1])
DipoleGroupWeight = np.ones(len(DipoleGroup))/np.float(len(DipoleGroup))


depth = None
cv_partition_ind = np.zeros(n_trials)
cv_partition_ind[1::2] = 1

# ==================================== the MNE solution =====================
from mne_stft_regression import (mne_stft_regression, \
                         select_lambda_tuning_mne_stft_regression_cv)
# create an inverse operator first
depth = None
inverse_operator = mne.minimum_norm.make_inverse_operator(evoked_list[0].info, 
                                                          fwd, cov, depth = depth,
                                                          fixed = True)                       
# tuning parameter range
snr_tuning_seq = np.array([0.5,1,2,3])
snr_tuning_star, cv_MSE = select_lambda_tuning_mne_stft_regression_cv\
                                            (evoked_list, inverse_operator,
                                                fwd, X,  cv_partition_ind,
                                                snr_tuning_seq, 
                                                labels = None, 
                                                wsize=wsize, tstep = tstep)
mne_result = mne_stft_regression(evoked_list, inverse_operator, X, 
                                   labels = labels, snr = snr_tuning_star,
                                   wsize = wsize, tstep = tstep)
Z_coef_mne = mne_result['coef']
# test an example sensor
#plt.imshow(np.abs(Z_coef_mne[1,:,:,3]))
#plt.show() 
#example_Z_coef = np.abs(Z_coef_mne[1,0,15,:])
#plt.figure()
#plt.subplot(1,2,1)
#plt.plot(x, np.dot(X,example_Z_coef))
#plt.title("MNE")
#plt.subplot(1,2,2)
#plt.plot(x, yhat*1E-9, 'g')
#plt.title("truth")
#plt.show()

# ====================================my method, get the correct gain matrix ==========
stc_mne = mne.minimum_norm.apply_inverse(evoked_list[-1], inverse_operator, 
                                         lambda2=1. / snr**2.,
                                          method='MNE')                   
thresh = 40E-10
active_set_z0 = np.mean(np.abs(stc_mne.data), axis = 1) > thresh
active_set_z0[true_active_set] = True
#active_set_z0 = np.zeros(n_dipoles,dtype = np.bool) 
print np.nonzero(active_set_z0)
print active_set_z0.sum()
Z0 = phi(stc_mne.data[active_set_z0,:])
active_t_ind0 = np.ones(25, dtype = np.bool)


from stft_tree_group_lasso.inverse_regression_stft_tsparse import *
import time

#alpha_seq = np.array([1E3,1E5,1E7])
#beta_seq = np.array([1E2,1E3])
alpha_seq = np.array([1E3,1E2])
beta_seq = np.array([5E2])
t0 = time.time()
Z_full, active_set, active_t_ind, stc_list, \
alpha_star, beta_star, gamma_star, cv_MSE_lasso, cv_MSE_L2 \
                 = get_stc_tree_group_lasso_regression_stft \
                                (evoked_list,X, fwd, cov,
                                labels, 1,
                                active_set_z0, active_t_ind0,
                                alpha_seq, beta_seq,
                                loose=0.2, depth=0.8, maxit=50, tol=1e-2,
                                wsize=wsize, tstep=4, window= 0.02, 
                                gamma_seq = np.array([1E-3]), L2_option = 1,
                                Flag_trial_by_trial = False)

coef_non_zero_mat = np.abs(Z_full)>0
Z_full_l2, active_set_l2, active_t_ind_l2, stc_list_l2, \
_, _, gamma_star, _, cv_MSE_L2 \
                 = get_stc_tree_group_lasso_regression_stft \
                                (evoked_list,X, fwd, cov,
                                labels, 1,
                                active_set, active_t_ind,
                                alpha_seq, beta_seq,
                                loose=0.2, depth=0.8, maxit=10, tol=1e-2,
                                wsize=wsize, tstep=4, window= 0.02, 
                                gamma_seq = np.array([1E-3,1E-4]), L2_option =2,
                                coef_non_zero_mat = coef_non_zero_mat, Z0_l2 = Z_full,
                                Flag_trial_by_trial = False)
print time.time()-t0


# ====================== comparison of the truth and MNE solutioins
# create a SourceEstimate object using the one trials
i0 = 10
vertices_0,_ = mne.source_space.label_src_vertno_sel(labels[0],fwd['src'])
vertices_1,_ = mne.source_space.label_src_vertno_sel(labels[1],fwd['src'])
vertices = [vertices_0[0].astype(np.int), vertices_1[1].astype(np.int)]
stc_true_trial_i = mne.SourceEstimate(data = true_stc_data_list[i0],
                                      vertices = vertices, tmin = -0.1,
                                      tstep = 0.01)
mne.viz.plot_sparse_source_estimates(fwd['src'], stc_true_trial_i)
# mne
mne_stc_trial_i_all = mne.minimum_norm.inverse.apply_inverse(evoked_list[i0],
                                     inverse_operator, lambda2 = 1.0/1**2,
                                            method = 'MNE') 
#mne_stc_trial_i = mne_stc_trial_i_all.in_label(label = labels[0]+labels[1])
mne_stc_trial_i = mne_stc_trial_i_all.in_label(label = labels[1])
mne.viz.plot_sparse_source_estimates(fwd['src'], mne_stc_trial_i)
# my method
#mne.viz.plot_sparse_source_estimates(fwd['src'], 
#                    stc_list[i0].in_label(label = labels[0]))
mne.viz.plot_sparse_source_estimates(fwd['src'], 
                   stc_list[i0])


time_id = 10 # 40
mne_data= mne_stc_trial_i.data[:, time_id]
my_data = stc_list[i0].data[:, time_id]
plt.plot(mne_data, my_data, '.')

# =============================================================                               
#plot_sparse_source_estimates(fwd['src'], stc_list[-1], bgcolor=(1, 1, 1),
#                             opacity=0.5, high_resolution=True)


active_set_ind = (np.nonzero(active_set))[0]
sel_ind = np.hstack([label_ind[0],label_ind[1]])
active_set_in_label = [i for i in range(len(active_set_ind)) \
                  if active_set_ind[i] in label_ind[0]  \
                      or active_set_ind[i] in label_ind[1]]


p = X.shape[1]
# my result      
Z_active_val = np.zeros(Z_true.shape,dtype = np.complex)    
true_active_set_in_active_set = [i for i in range(len(sel_ind)) \
              if sel_ind[i] in active_set_ind ]
Z_active_val[true_active_set_in_active_set] \
        = Z[active_set_in_label,0:n_coefs*p] 
                     
# mne result
sel_ind = np.hstack([label_ind[0],label_ind[1]])
Z_active_mne_val = np.zeros([len(sel_ind), p*n_coefs], dtype = np.complex)
for i in range(len(sel_ind)):
    for l in range(p):
        Z_active_mne_val[i,l*n_coefs:(l+1)*n_coefs] = np.ravel(Z_coef_mne[i,:,:,l], order = 'C')
    

vmin = 0.0
vmax = 5E-8
plt.figure()
plt.subplot(3,1,1)
plt.imshow(np.abs(Z_active_val),vmin= vmin, vmax = vmax)
#plt.colorbar()
plt.title('my method')
plt.subplot(3,1,2)
plt.imshow(np.abs(Z_active_mne_val),vmin= vmin, vmax = vmax)
#plt.colorbar()
plt.title('mne regression')
plt.subplot(3,1,3)
plt.imshow(np.abs(Z_true),vmin= vmin, vmax = vmax)
#plt.colorbar()
plt.title('truth mean')
plt.show()


coef_error_MNE = np.sum(np.abs(Z_active_mne_val-Z_true))
coef_error = np.sum(np.abs(Z_active_val-Z_true))


# ========= also check the error on the curve? =============================
curve_error_MNE, curve_error = 0.0,0.0
for i in range(Z_true.shape[0]):
    for j in range(n_coefs):
        # truth
        true_curve_real = X.dot(np.real(Z_true[i, j+np.arange(0,p)*n_coefs]))
        true_curve_imag = X.dot(np.imag(Z_true[i, j+np.arange(0,p)*n_coefs]))
        # MNE
        mne_curve_real = X.dot(np.real(Z_active_mne_val[i, j+np.arange(0,p)*n_coefs]))
        mne_curve_imag = X.dot(np.imag(Z_active_mne_val[i, j+np.arange(0,p)*n_coefs]))
        # my method
        curve_real = X.dot(np.real(Z_active_val[i, j+np.arange(0,p)*n_coefs]))
        curve_imag = X.dot(np.imag(Z_active_val[i, j+np.arange(0,p)*n_coefs]))
        # the total error
        curve_error_MNE += ((true_curve_real-mne_curve_real)**2).sum() \
                          + ((true_curve_imag-mne_curve_imag)**2).sum()
        curve_error += ((true_curve_real-curve_real)**2).sum() \
                          + ((true_curve_imag-curve_imag)**2).sum()
        
        plt.figure()
        plt.plot(x,true_curve_real,'g')
        plt.plot(x,mne_curve_real,'r')
        plt.plot(x,curve_real, 'b')
        plt.legend(("truth", "mne","my method"),loc = 4)
        plt.show()

# ==================== the error in source space, only on the active set=====
estimated_stc_data = mne_result['roi_data_3D']
MSE_source = 0.0
MSE_source_mne = 0.0
for i in range(n_trials):
    tmp_stc_my_method = np.zeros(true_stc_data_list[i].shape)
    tmp_stc_my_method[true_active_set_in_active_set,:] = stc_list[i].data[active_set_in_label,:]
    MSE_source += np.sum((tmp_stc_my_method-true_stc_data_list[i])**2)
    MSE_source_mne += np.sum((estimated_stc_data[:,:,i] - true_stc_data_list[i])**2)

MSE_source /= n_trials
MSE_source_mne /= n_trials


# ==================== 
plt.figure()
ax = plt.subplot(3,1,1)
plt.bar(np.array([0,1]), np.array([coef_error_MNE, coef_error]))
ax.set_xticks([0,1])
ax.set_xticklabels(['MNE','my method'])
plt.ylabel('error of coefficients')
ax = plt.subplot(3,1,2)
plt.bar(np.array([0,1]), np.array([curve_error_MNE, curve_error]))
ax.set_xticks([0,1])
ax.set_xticklabels(['MNE','my method'])
plt.ylabel('error of curves')
ax = plt.subplot(3,1,3)
plt.bar(np.array([0,1]), np.array([MSE_source_mne, MSE_source]))
ax.set_xticks([0,1])
ax.set_xticklabels(['MNE','my method'])
plt.ylabel('source_MSE')
plt.show()    

# =================
i = 19
plt.figure()
ax = plt.subplot(2,2,1)
plt.plot(stc_list[i].data[active_set_in_label,:].T)
plt.title('my method')
ax = plt.subplot(2,2,2)
plt.plot(estimated_stc_data[:,:,i].T)
plt.title('MNE')
ax = plt.subplot(2,2,3)
plt.plot(true_stc_data_list[i].T)
plt.title('Truth')
plt.show()                          
