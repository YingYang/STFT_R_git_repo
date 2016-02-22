import matplotlib.pyplot as plt
import numpy as np
from mne.inverse_sparse.mxne_optim import _Phi, _PhiT
import os,sys,inspect
# for my desktop
os.chdir('/home/ying/Dropbox/MEG_source_loc_proj/STFT_R_git_repo/')
# for my laptop
#os.chdir('/home/yingyang/Dropbox/MEG_source_loc_proj/stft_tree_group_lasso/')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import STFT_R
import time
from STFT_R.get_gradient import get_gradient0, get_gradient1, get_gradient1_L21
 

n_channels = 30
n_times = 100
n_dipoles = 1000
G = np.random.randn(n_channels, n_dipoles)
n0 = 6
true_group = np.arange(0,n0)

# note the elements of the list must be arrays
n_orient = 1

DipoleGroup = list()
for i in range(n_dipoles):
    DipoleGroup.append(np.array([i]).astype(int))
    
DipoleGroupWeight = np.ones(len(DipoleGroup))/np.float(len(DipoleGroup))

#alpha, beta = 1E3*n_dipoles,1E-2*n_dipoles
alpha0, beta0, gamma0 = 10,10,10
# test the dual funciton of tfMxNM
n_times = 100
wsize, tstep = 16,4
n_step = int(np.ceil(n_times/float(tstep)))
n_freq = wsize// 2+1
n_coefs = n_step*n_freq
phi = _Phi(wsize, tstep, n_coefs)
phiT = _PhiT(tstep, n_freq, n_step, n_times)

n_trials = 20
X = np.arange(1, n_trials+1,1)
M = np.zeros([n_channels, n_times, n_trials])
source_signal = np.zeros([n_dipoles, n_times, n_trials])
m0 = len(true_group)
snr = 2

times = np.arange(0,n_times)
time_course =(np.sin(times*0.1) + np.cos(times*0.9))* 2*np.exp(-(times-20)**2/1E2) 
    #+ 0.1*rd.randn(len(times))
#time_course =  10*np.exp(-(times-50)**2/1E3) 
Z00 = phi(time_course)
fig = plt.figure()
fig.add_subplot(2,1,1)
plt.plot(times, time_course)
plt.xlabel('time')
plt.title('time series')
fig.add_subplot(2,1,2)
plt.imshow(np.abs(phi(time_course)).reshape([n_freq,n_step]))
plt.title('absolute value of STFT coeficient')
plt.show()
plt.close('all')
Z00 = np.tile(Z00, [m0,1])
Z00 = np.hstack([Z00,Z00])

dm = np.vstack([np.ones(len(X))/10.0,(X-10.0)/20.0]).T
#Z00_true = np.zeros([n_dipoles, n_coefs], dtype = complex)
#Z00_true[true_group,:] = Z00[:,0:n_coefs]
Z00[0,:] *= 0.01;
for i in range(n_trials):
    #tmpZ = Z0* snr*X[i]*(1+1j) + 1.0/snr*(rd.randn(m0,n_coefs) + 1j* rd.randn(m0,n_coefs))
    tmpZ = Z00[:,0:n_coefs]*dm[i,0] + Z00[:,n_coefs:2*n_coefs]*dm[i,1]
    Z = np.zeros([n_dipoles,n_coefs], dtype = complex)
    Z[true_group,:] = tmpZ
    source_signal[:,:,i] = phiT(Z)
    signal = phiT(G.dot(Z))
    #signal2 = G.dot(source_signal[:,:,i])
    M[:,:,i] = signal + 1/snr*np.random.randn(n_channels, n_times)

## ==============================================tfMxNE =================================================  
#maxit = 300
#M1 = M[:,:,0]
#import mne
#alpha0 = 5
#beta0 = 5
#source_data,active_set_tfMxNE, E = mne.inverse_sparse.mxne_optim.tf_mixed_norm_solver(
#                         M1,G, alpha_space = alpha0, alpha_time = beta0, 
#                          wsize = wsize, tstep = tstep, maxit = 500, 
#                          debias = False)
#Z_tfMxNE = phi(source_data)
#z_dual = G[:, active_set_tfMxNE].dot(source_data)
#u = z_dual - M1
#dual_obj = -0.5*(np.sum(z_dual**2) - np.sum(M1**2))
#print dual_obj, E[-1],  E[-1] - dual_obj
#gradient = phi(G.T.dot(u))
#abs_grad = np.abs(gradient)

# my method
p = 2
DipoleGroupWeight = np.ones(len(DipoleGroup))/np.float(len(DipoleGroup))
alpha = 1E-1*n_dipoles
beta = 1E-1*n_dipoles
Z_ini = (np.random.randn(n_dipoles,n_coefs*p) + 1j* np.random.randn(n_dipoles,n_coefs*p))*1E-3
active_set_z_ini = np.ones(n_dipoles, dtype = np.bool)
active_t_ind_ini = np.ones(n_step, dtype = np.bool)

alpha, beta, gamma = 10,10,10
X = dm.copy()            
G_list = [G]
G_ind = np.zeros(X.shape[0], dtype = np.int)    

maxit = 100
tol = 1E-3
t0 = time.time()         
Z,active_set, obj = STFT_R.L21_tsparse.solve_stft_regression_tree_group_tsparse\
                                (M, G_list, G_ind, X, alpha,beta, gamma, 
                                DipoleGroup,DipoleGroupWeight,
                                Z_ini, active_set_z_ini, 
                                n_orient=1, wsize=16, tstep = 4,
                                maxit=maxit, tol = tol,lipschitz_constant = None,
                                Flag_backtrack = True, eta = 1.5, L0 = 1.0,
                                Flag_verbose = True)
time0 = time.time()-t0

if False:
    t0 = time.time()
    Z1,active_set1, obj1 = STFT_R.L21_tsparse.solve_stft_regression_tree_group_tsparse\
                                    (M, G_list, G_ind, X, alpha,beta, gamma, 
                                    DipoleGroup,DipoleGroupWeight,
                                    Z_ini, active_set_z_ini, 
                                    n_orient=1, wsize=16, tstep = 4,
                                    maxit=maxit, tol = tol,lipschitz_constant = None,
                                    Flag_backtrack = False, eta = 1.5, L0 = 1.0,
                                    Flag_verbose = True)                                                           
    time1 = time.time()-t0
    
    # The following are tested, but now disabled. L21 will no longer be imported. 
#    t0 = time.time()         
#    Z2,active_set2, obj2 = STFT_R.L21.solve_stft_regression_tree_group\
#                                    (M, G_list, G_ind, X, alpha,beta, gamma, 
#                                    DipoleGroup,DipoleGroupWeight,
#                                    Z_ini, active_set_z_ini, 
#                                    n_orient=1, wsize=16, tstep = 4,
#                                    maxit=maxit, tol = tol,lipschitz_constant = None,
#                                    Flag_backtrack = True, eta = 1.5, L0 = 1.0,
#                                    Flag_verbose = True)
#    time2= time.time()-t0
#    t0 = time.time()         
#    Z3,active_set3, obj3 = STFT_R.L21.solve_stft_regression_tree_group\
#                                    (M, G_list, G_ind, X, alpha,beta, gamma, 
#                                    DipoleGroup,DipoleGroupWeight,
#                                    Z_ini, active_set_z_ini, 
#                                    n_orient=1, wsize=16, tstep = 4,
#                                    maxit=maxit, tol = tol,lipschitz_constant = None,
#                                    Flag_backtrack = False, eta = 1.5, L0 = 1.0,
#                                    Flag_verbose = True)
#    time3 =  time.time()-t0
#    print np.linalg.norm(Z2-Z), np.linalg.norm(Z3-Z1)


coef_non_zero_mat = np.abs(Z) >0
t0 = time.time()
Z_L2, obj_L2 = STFT_R.L2_tsparse.solve_stft_regression_L2_tsparse(M,G_list, G_ind, X, Z, 
                                    active_set, np.ones(n_step, dtype = np.bool),
                                    coef_non_zero_mat,
                                wsize=16, tstep = 4, delta = 0,
                                maxit=100, tol = 1e-3,lipschitz_constant = None,
                                Flag_backtrack = True, eta = 1.2, L0 = 1,
                                Flag_verbose = True)
print time.time()-t0
if False:
    # visualization
    # My estimate can be larger than the truth,
    # In lasso, theoretically, the estimated is biased towards zero
    # Yifei did an experiment, if the design matrix is has a large cond val,
    # i.e. low rank, this can happen. 
    # my design matrix are the STFT coefs. so yes. 
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(np.real(Z).T)
    plt.subplot(2,2,2)
    plt.plot(np.real(Z00).T)
    plt.subplot(2,2,3)
    plt.plot(np.imag(Z).T)
    plt.subplot(2,2,4)
    plt.plot(np.imag(Z00).T)
    plt.show()
    
    Z, active_set, obj = STFT_R.L21_tsparse.solve_stft_regression_tree_group_tsparse\
                                (M, G_list, G_ind, X, alpha,beta, gamma, 
                                DipoleGroup,DipoleGroupWeight,
                                Z_ini, active_set_z_ini, 
                                n_orient=1, wsize=16, tstep = 4,
                                maxit=500, tol =1E-6,lipschitz_constant = None,
                                Flag_backtrack = False, eta = 2.0, L0 = 1.0,
                                Flag_verbose = True)
    
    result_dual = STFT_R.L21_duality.compute_dual_gap( M, G_list, G_ind, X, Z, active_set, 
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = 16, tstep = 4)
    print result_dual['feasibility_dist']/np.linalg.norm(result_dual['gradient'])
    print result_dual['feasibility_dist']/np.linalg.norm(result_dual['b'])

    active_set_z = np.zeros(n_dipoles, dtype = bool)
    active_set_z[0:6] = True
    Z1,active_set_z1,obj1 = STFT_R.L21_tsparse.solve_stft_regression_tree_group_tsparse\
                                    (M, [G[:,true_group]], np.zeros(n_trials, dtype = int),
                                     X, alpha,beta, gamma, 
                                 DipoleGroup[0:6],DipoleGroupWeight[0:6],
                                    Z_ini[0:6,:], active_set_z[0:6], 
                                n_orient=1, wsize=16, tstep = 4,
                                maxit=maxit, tol = tol,lipschitz_constant = None,
                                Flag_backtrack = True, eta = 1.1, L0 = 1.0,
                                Flag_verbose = True)
                               
    active_set_z2 = np.zeros(n_dipoles, dtype = np.bool)
    active_set_z2[0:6]= active_set_z1    
    result_dual = STFT_R.L21_duality.compute_dual_gap(
                    M, G_list, G_ind, X, Z1, active_set_z2, 
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = 16, tstep = 4)
    print result_dual['feasibility_dist']/np.linalg.norm(result_dual['gradient'])
    # more tests
    active_set0 = np.zeros(n_dipoles, dtype = np.bool)
    active_set0[true_group] = True
    result_dual = STFT_R.L21_duality.compute_dual_gap(
                    M, G_list, G_ind, X, Z00, active_set0, 
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = 16, tstep = 4)
    print result_dual['feasibility_dist']/np.linalg.norm(result_dual['gradient'])
    
    
    Z_ini1 = Z_ini[7:10]
    active_set_ini1 = np.zeros(n_dipoles, dtype = np.bool)
    active_set_ini1[7:10] = True
    result_dual = STFT_R.L21_duality.compute_dual_gap(
                    M, G_list, G_ind, X, Z_ini1, active_set_ini1, 
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = 16, tstep = 4)
    print result_dual['feasibility_dist']/np.linalg.norm(result_dual['gradient'])
    plt.plot(result_dual['feasibility_dist_DipoleGroup'])
    
    active_set_J_ini = np.zeros(len(DipoleGroup), dtype = np.bool)
    active_set_J_ini[0:1] = True
    active_set_J_ini[15:20] = True
    
    result = STFT_R.L21_active_set.solve_stft_regression_tree_group_active_set\
                                    (M,G_list, G_ind, X,
                                alpha,beta, gamma, DipoleGroup,DipoleGroupWeight,
                                Z_ini,active_set_z_ini, 
                                active_set_J_ini, 
                                n_orient=1, wsize= 16, tstep = 4,
                                maxit=700, tol = 1e-6,
                                Maxit_J = 10, Incre_Group_Numb = 50, dual_tol = 1e-2,
                                Flag_verbose = False,
                                Flag_backtrack = False, L0 = 1.0, eta = 1.01)
    
    result_dual = STFT_R.L21_duality.compute_dual_gap(
                    M, G_list, G_ind, X, result['Z'], result['active_set'],  
              alpha, beta, gamma,
              DipoleGroup, DipoleGroupWeight, n_orient,
              wsize = 16, tstep = 4)          
    print result_dual['feasibility_dist']/np.linalg.norm(result_dual['gradient'])
    
    alpha_seq = np.array([1E2, 1E1])
    beta_seq = np.array([1E1])
    gamma_seq = np.array([1E1])
    cv_partition_ind = np.zeros(n_trials)
    cv_partition_ind[0::2] = 1
    alpha_star, beta_star, gamma_star, cv_MSE = STFT_R.L21_active_set.select_alpha_beta_gamma_stft_tree_group_cv_active_set\
                                           (M,G_list, G_ind,X, 
                                        active_set_J_ini, 
                                         DipoleGroup,DipoleGroupWeight,
                                         alpha_seq, beta_seq, gamma_seq,
                                         cv_partition_ind,
                                         n_orient=1, wsize= 16, tstep = 4, 
                                         maxit=500, tol = 1e-4,
                                         Maxit_J = 10, Incre_Group_Numb = 50, 
                                         dual_tol = 1e-2,
                                         Flag_verbose = False,
                                         Flag_backtrack = False, L0 = 1.0, eta = 1.1)
