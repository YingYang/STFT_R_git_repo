"""
1. Create simulation data based on MNE-python sample data set, 
The time series of source signals in each trial in some regions 
of interest (ROIs) are correlated with the covariate or regressor(X).
We assume the the all the source points in each ROI have common regression
coefficents, but there is 

2. Use the STFT-R and two-step MNE+regrssion to estimate the regression 
coefficients. Then visualize the results. 
"""

# ========================= 0. Add paths and import modules ===================
# NOTE: if you need to apply either STFT-R or MNE-R on your own data, please 
# add the absolute path of "STFT_R", "MNE_stft","evaluation", and then import the modules
# then import the Evaluation_individual_G  

import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
mne.set_log_level('warning')


# adding all the required modules in the path

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
print "current_dir %s" % current_dir
print "parent_dir %s" % parent_dir
print os.path.dirname(parent_dir)
sys.path.insert(0,parent_dir)
sys.path.insert(0,parent_dir + '/evaluation/')
sys.path.insert(0,parent_dir + '/MNE_stft/')
from Evaluation_individual_G import (get_solution_individual_G,
                        get_bootstrapped_solution_individual_G)

#============================1. Create the simulation data ====================
# this part was modified from 
# number of trials 
n_trials = 40
# a simple regressor, with two columns: intercept and slope
X = np.hstack([np.ones([n_trials, 1]), np.linspace(-0.5,0.5, num = n_trials)[:, np.newaxis]])
# p is the dimension of X
p = X.shape[1]
# use the pre-defined ROIs from MNE-python sample data set, 
# two primary auditory regions and two primary visual regions
label_names = ['Aud-lh','Aud-rh','Vis-lh','Vis-rh']
ROI_path = "%s/Simulation_Sample_Subject_ROIs/" %current_dir
ROI_file_paths = [(ROI_path + '%s.label' % ln)  for ln in label_names]
n_ROI = len(ROI_file_paths) 
labels = [mne.read_label(ROI_file_paths[ln]) for ln in range(n_ROI)]

# load sample data to create noise of the sensor 
data_path = mne.datasets.sample.data_path()   
# load raw data, add SSP projections 
raw = mne.io.Raw(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg_proj.fif')
raw.info['projs'] += proj
# exclude the bad channels
bads = ['MEG 2443', 'EEG 053']
raw.info['bads'] = bads  # mark bad channels 
# For simplicity, we only use MEG sensors here. 
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
# compute the sensor noise covariance matrix from the raw data
cov = mne.cov.compute_raw_covariance(raw, tmin = 20.0, tmax = 100.0, picks = picks)

# load evoked averaged data in one condition, as a template, where we can put 
# our simulated data in. 
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
evoked_template = mne.read_evokeds(ave_fname)[0]
evoked_template.pick_types(meg=True, eeg=False,exclude= bads)
# then the raw data can be removed from the memory
del(raw)

# create the mean time courses of each ROI
n_times = 100
stc_data = np.zeros([4,n_times])
sfreq = 100.0
Ws = mne.time_frequency.morlet(sfreq, [3, 10], n_cycles=[1, 1.5])
stc_data[0][:len(Ws[0])] = np.real(Ws[0])
stc_data[2][:len(Ws[0])] = np.real(Ws[0])
stc_data[1][:len(Ws[0])] = np.real(Ws[0])
stc_data[3][:len(Ws[0])] = np.real(Ws[0])
# time translation
stc_data[1] = np.roll(stc_data[1], 25)
stc_data[2] = np.roll(stc_data[2], 16)
stc_data[3] = np.roll(stc_data[3], 10)

# covariance of the temporal noise for each source point
stc_covar = np.zeros([n_times, n_times], dtype = np.float)
a = 1e-3
for i in range(n_times):
    for j in range(n_times):
        stc_covar[i,j] = np.exp(- a*(i-j)**2)
        if i == j:
            stc_covar[i,j] *= 1.01

amp = 10.0*1e-9
stc_data *= amp  # use nAm as unit

# load the forward solution
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False, exclude=bads)
# get the indices of source points for each ROI
src = fwd['src']
label_ind = list()
for i in range(len(labels)):
    _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
    label_ind.append(sel)

# stc0_data: source time series template with no noise        
n_source = fwd['sol']['ncol']
stc0_data = np.zeros([n_source, n_times])
for i in range(len(labels)):
    stc0 =  mne.simulation.generate_stc(fwd['src'], labels[i:i+1], stc_data[i:i+1,:], -0.1, 1.0/sfreq)
    stc0_data[label_ind[i],:] = stc0.data
    del(stc0)
            
# get the active set (all source points in the ROIs)
active_set = np.zeros(0)
for i in range(len(labels)):
    active_set = np.hstack([active_set, label_ind[i]])    
active_set = np.unique(active_set)
active_set = active_set.astype(np.int)
n_valid_source = len(active_set)

# parameters for STFT-transform : 
# tstep_phi: time steps of STFT-R
# wsize_phi: window size of the STFT-R              
tstep_phi = 4
wsize_phi = 16  
# number of time steps
n_step = int(np.ceil(n_times/float(tstep_phi)))
# number of frequencies
n_freq = wsize_phi// 2+1
# n_coefs, total number of time-frequency components
n_tfs = n_step*n_freq   
#STFT and inverse STFT function from MNE-python
phi = _Phi(wsize_phi, tstep_phi, n_tfs)
phiT = _PhiT(tstep_phi, n_freq, n_step, n_times)                          
                         
#Z_true: regression coeffients in the time-frequency domain
coef_per_ROI = np.ones([n_ROI,n_freq,n_step, p]) 
Z_true = np.tile(phi(stc0_data),[1,p])
for i in range(len(labels)):
    Z_true[label_ind[i]] *= np.reshape(np.transpose(coef_per_ROI[i],[2,0,1]),[1,-1]) 

# create the sensor data and source data for each trial        
evoked_list = list()
stc_data_list = list()
label_all = labels[0]
for i in range(1,len(labels)):
    label_all += labels[i]
    
# SNR of the sensors: signal here is the source time series
sensor_snr =1.0
# noise_ratio_source: the ratio between the std of the Gaussian noise source 
# and the largest absolute value of the signal
# this is an array, including a distinct value for each ROI, 
# the final element cooresponds to all sources outside the ROIs
noise_ratio_source = np.array([0.1, 0.1,0.1,0.1, 0.05])
vertices = [src[0]['vertno'], src[1]['vertno']]
for r in range(n_trials):
    tmp_stc_data  = stc0_data.copy()
    for i in range(len(labels)):
        tmpZ = np.swapaxes(np.reshape(Z_true[label_ind[i][0],:],[p,-1]),0,1)
        tmpZ = np.sum(tmpZ*X[r,:],1)
        tmp_stc = phiT(tmpZ)
        tmp_stc_data[label_ind[i],:] = np.tile(tmp_stc, (len(label_ind[i]),1))
        # the GP_noise for each trial should be i.i.d.
        tmp_GP_noise = np.random.multivariate_normal(np.zeros(n_times, dtype= np.float),
                                     stc_covar,len(label_ind[i]))
        tmp_stc_data[label_ind[i],:] += tmp_GP_noise*noise_ratio_source[i]*amp

    # adding GP noise
    ind_non_active = np.setdiff1d(np.arange(0, n_source), active_set)
    tmp_GP_noise = np.random.multivariate_normal(np.zeros(n_times, dtype= np.float),
                                     stc_covar,len(ind_non_active))
    tmp_stc_data[ind_non_active,:] += tmp_GP_noise*noise_ratio_source[-1]*amp
    
    stc_data_list.append(tmp_stc_data[:,:])                         
    stc = mne.SourceEstimate(data = tmp_stc_data,vertices = vertices,
                             tmin = -0.1, tstep = 1.0/sfreq )  
    # generate evoked data
    evoked = mne.simulation.simulate_evoked(fwd, stc, evoked_template, cov, sensor_snr,
                     tmin=0.0, tmax=0.2, iir_filter= None)
    evoked_list.append(evoked)
    # save all the parameters above in a dict object. 
    simu_obj = dict(evoked_list = evoked_list, X = X,
                    fwd = fwd, stc_data = stc_data,
                    stc_data_list = stc_data_list,
                    active_set = active_set,
                    labels = labels,  label_ind = label_ind, 
                    cov = cov, sensor_snr = sensor_snr, noise_ratio_source = noise_ratio_source,
                    Z_true = Z_true[active_set,:], tmp_stc = stc)

# debug:
print len(evoked_list)

#===================2. estimate the regression coefficients ====================  

label_ind0 = simu_obj['label_ind']
labels0 = simu_obj['labels']
n_source = fwd['sol']['data'].shape[1]
n_times = simu_obj['stc_data_list'][0].shape[1] 
active_ind = simu_obj['active_set']
n_coefs_all = simu_obj['Z_true'].shape[1]
Z_true = np.zeros([n_source,n_coefs_all ], dtype = np.complex)
Z_true[active_ind,:] = simu_obj['Z_true']
n_trials_all = 40
n_trials = 20

# treat two regions as real ROIs, two as irrelevant distractors
n_ROI_true = 2
label_ind = label_ind0[0:n_ROI_true]
labels = labels0[0:n_ROI_true]

X[:,1:] -= np.mean(X[:,1:])  
    
#split the trials for learning the L21 structure (training)
#and doing statistical inference (held)
hold_ind = np.arange(0,n_trials_all,2)
train_ind = np.arange(1,n_trials_all,2)
X_hold = X[hold_ind,:]
X_train = X[train_ind,:]
evoked_list_train = [evoked_list[i] for i in train_ind]
evoked_list_hold = [evoked_list[i] for i in hold_ind]


# reconstruct the true signal
n_times = evoked_list[0].data.shape[1]                 
phiT = _PhiT(tstep_phi, n_freq, n_step, n_times)
# use the true signal without noise as the truth
reshape_Z_true = np.reshape(simu_obj['Z_true'], [len(active_ind), p, -1])
reshape_Z_true = np.swapaxes(reshape_Z_true, 1,2)
true_stc_data_list = list()
# in this case, the evaluation function assumes true_stc_data_list to be full 
for r in range(n_trials):
    tmp_data = phiT( np.sum( reshape_Z_true * X_train[r,:],  axis = 2 ))
    tmp_full_data = np.zeros([n_source, n_times])
    tmp_full_data[active_ind,:] = tmp_data
    true_stc_data_list.append(tmp_full_data)


# searching this will take a long time, for demonstration, we reduce the searching rage
#tuning paramters for the L21, grid search on alpha, beta, gamma, 
alpha_seq = np.exp(np.arange(6,4,-1))*10.0
beta_seq = np.exp(np.arange(3,0,-1))*2.0
gamma_seq = np.exp(np.arange(0,-1,-1))
# L_2 tuning parameter
delta_seq = np.exp(np.arange(-6,2,1))
# tuning paramter ("snr" parameter in MNE) 
snr_tuning_seq = np.array([0.5,1,2])

# the evalutation code can accepted different fwd solutions for different trials 
# described by (fwd_list and G_ind) 
# here, we only used one single forward solution. 
fwd_list = list([fwd])
G_ind = np.zeros(len(evoked_list_train), dtype = np.int)

maxit = 10
tol = 1e-2
# =======STFT-R L21 solution, on the training set ================
result_STFT_R1 = get_solution_individual_G(evoked_list_train, fwd_list, G_ind, cov, X_train, labels, label_ind,
                          alpha_seq, beta_seq, gamma_seq,
                          delta_seq = delta_seq, snr_tuning_seq = snr_tuning_seq,
                          wsize = wsize_phi, tstep = tstep_phi, maxit = maxit, tol = tol,
                           method = "STFT-R", Incre_Group_Numb= 150, L2_option = 0, ROI_weight_param=0, 
                          Maxit_J = 2, dual_tol = 0.15)

plt.figure()                          
plt.imshow( np.abs(result_STFT_R1['Z']), aspect = "auto", interpolation = "none")

# =========MNE-R solution, on the training set
result_MNE_R1 = get_solution_individual_G(evoked_list_train, fwd_list,  G_ind, cov, X_train, labels, label_ind,
                          alpha_seq = None, beta_seq = None, gamma_seq = None,
                          delta_seq = None,
                          snr_tuning_seq = snr_tuning_seq,
                          wsize = wsize_phi, tstep = tstep_phi,method = "MNE-R")  


# ============ visulize the estimate on the training set ==========
Z_full = np.zeros([n_source, n_coefs_all], dtype = np.complex)
Z_full[result_STFT_R1['active_set'],:] =result_STFT_R1['Z'] 
Z_mne = result_MNE_R1['Z']

# show the absolute values of the regression coefficients 
vmin = 0
vmax = 1E-9
# all dipoles
plt.figure()                            
plt.subplot(3,1,1)  
plt.imshow( np.abs(Z_true), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax)
plt.colorbar()
plt.title("truth")            
plt.subplot(3,1,2)                      
plt.imshow( np.abs(Z_full), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar() 
plt.title("STFT-R")     
plt.subplot(3,1,3)                      
plt.imshow( np.abs(Z_mne), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar() 
plt.title("MNE-R")            

vmin = 0
vmax = 1E-9
# the active set only
plt.figure()                            
plt.subplot(3,1,1)  
plt.imshow( np.abs(Z_true[active_ind,:]), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax)
plt.colorbar() 
plt.title("truth")          
plt.subplot(3,1,2)                      
plt.imshow( np.abs(Z_full[active_ind,:]), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar()  
plt.title("STFT-R") 
plt.subplot(3,1,3)                      
plt.imshow( np.abs(Z_mne[active_ind,:]), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar()   
plt.title("MNE-R")


## L2 solution on the held data,    
G_ind = np.zeros(len(evoked_list_hold), dtype = np.int)
# STFT-R
result_STFT_R2 = get_solution_individual_G(evoked_list_hold, fwd_list, G_ind, cov, X_hold, 
                                           labels, label_ind,
                          alpha_seq, beta_seq, gamma_seq,
                          delta_seq = delta_seq, snr_tuning_seq = snr_tuning_seq,
                          wsize = wsize_phi, tstep = tstep_phi, maxit = maxit,tol = tol,
                           method = "STFT-R",
                          Incre_Group_Numb= 150, L2_option = 2, ROI_weight_param=0, 
                          active_set = result_STFT_R1['active_set'],
                          coef_non_zero_mat = result_STFT_R1['coef_non_zero_mat'])
# MNE-R        
result_MNE_R2 = get_solution_individual_G(evoked_list_hold, fwd_list, G_ind, cov, X_hold, 
                                          labels, label_ind,
                          alpha_seq = None, beta_seq = None, gamma_seq = None,
                          delta_seq = None,
                          snr_tuning_seq = snr_tuning_seq,
                          wsize = wsize_phi, tstep = tstep_phi,method = "MNE-R")



### To be updated!!!!    
path = current_dir
B = 5
# note: this will take very long                      
result_STFT_R2btstrp = get_bootstrapped_solution_individual_G(evoked_list_hold, 
                            fwd_list, G_ind, cov, X_hold, result_STFT_R2['Z'], 
                            result_STFT_R2['active_set'], 
                            result_STFT_R2['coef_non_zero_mat'], 
                            path, "STFT", labels, label_ind, 
                            method='STFT-R', maxit=maxit, tol=tol, 
                            B=B, wsize=wsize_phi, tstep=tstep_phi, 
                            Rescale_Flag=True, delta_seq= delta_seq, 
                            snr_tuning_seq=None, depth=None)


result_MNE_R2btstrp =get_bootstrapped_solution_individual_G(evoked_list_hold, 
                            fwd_list, G_ind, cov, X_hold, result_MNE_R2['Z'], 
                            result_MNE_R2['active_set'], 
                            result_MNE_R2['coef_non_zero_mat'], 
                            path, "MNE", labels, label_ind, 
                            method='MNE-R',
                            B=B, wsize=wsize_phi, tstep=tstep_phi, 
                            Rescale_Flag=True, delta_seq= None, 
                            snr_tuning_seq= snr_tuning_seq, depth=None)



T_active_STFT = np.abs(result_STFT_R2['Z'])/np.std( np.abs(result_STFT_R2btstrp['Z_btstrp']), axis = 2)
T_active_STFT[np.isnan(T_active_STFT)] = 0
T_full = np.zeros(Z_full.shape)
T_full[result_STFT_R2['active_set']] = T_active_STFT

T_MNE = np.abs(result_MNE_R2['Z'])/np.std( np.abs(result_MNE_R2btstrp['Z_btstrp']), axis = 2)
T_MNE[np.isnan(T_MNE)] = 0


# show the T-statistics of Absolute values of the regression coefficients  
plt.figure()
vmin = 0
vmax = 40                            
plt.subplot(1,3,1) 
plt.imshow(np.abs(Z_true), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = None)
plt.colorbar() 
plt.title("Truth")
plt.xlabel("time")
plt.ylabel("source point index")          
plt.subplot(1,3,2)                      
plt.imshow( T_full, aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar()  
plt.title("STFT-R")
plt.xlabel("time")
plt.ylabel("source point index") 
plt.subplot(1,3,3)                      
plt.imshow( T_MNE, aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar() 
plt.title("MNE-R") 
plt.xlabel("time")
plt.ylabel("source point index")            


# the active set only
plt.figure()                            
plt.subplot(3,1,1)  
plt.imshow( np.abs(Z_true[active_ind,:]), aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = None)
plt.colorbar() 
plt.title("Truth, all active sources")
plt.xlabel("time")
plt.ylabel("source point index")           
plt.subplot(3,1,2)                      
plt.imshow( T_full[active_ind,:], aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar()   
plt.xlabel("time")
plt.title("STFT-R, all active sources")
plt.ylabel("source point index") 
plt.subplot(3,1,3)                      
plt.imshow( T_MNE[active_ind,:], aspect = "auto", interpolation = "none",
           vmin = vmin, vmax = vmax) 
plt.colorbar()
plt.title("MNE-R, all active sources")
plt.xlabel("time")
plt.ylabel("source point index")   








