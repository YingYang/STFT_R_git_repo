# -*- coding: utf-8 -*-
"""
compute the proximal operator for 1/2||y-z||_2^2 + \alpha Omega(z)
for complex variable z,
using the Algorithm 2 in Jenatton et al., 2011

note that this takes a full matrix as an input, 
the # rows in y, and the dipole groups must match. 
the code is sped up by using vector/matrix operations
"""

import numpy as np
#from mne.inverse_sparse.mxne_optim import *

def prox_tree_hard_coded_full_matrix(y, n_coefs, p, 
              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient):
    """ proximal operator for a tree-structured group, 
    the input y and the output Z are complex arrays
    the lower level grouping structure is hardcoded
    
    Input: 
        y, [n_dipoles, n_coefs*p] complex matrix, order, [n_dipoles, pq,n_coefs]
        n_coefs, number of coefficients after stft transform
        p, number of covariates       
        alpha, real constant, the penalize parameter for the dipole groups
        beta, the penalization on each group of coefficients
        gamma, the penalization on each single entry of Z
        DipoleGroup, a list of index sets,  the grouping of dipoles
        GroupWeight, weight of each group, sum to one
        n_orient, number of orientations, 1 or 3,  
                  if 3, neighboring 3 rows are the same dipole
                  make sure DipoleGroup contain all the 3 components of one dipole            
    Output:
        z, n_dipoles(true) x n_coefs x p, complex array
        active_set_z, active_set, (n_dipole,) boolean array
    """ 
    # initialization
    z = y.copy()
    n_dipoles = y.shape[0]
    if y.shape[1] != n_coefs*p:
        raise ValueError("the number of coefficients do not match")
    # =========level 0, absolute value of coefficients======================
    if n_orient == 1: 
        l1_norm = np.abs(z)
        l1_norm[l1_norm <= gamma] = gamma
        shrink_mat = 1.0 - gamma/l1_norm
        shrink_mat[shrink_mat<=0] = 0.0
        z *= shrink_mat
    elif n_orient == 3:
        reshape_z = np.reshape(z, [n_dipoles//3,3,-1])
        l1_norm = np.sqrt(np.sum(np.abs(reshape_z)**2, axis = 1))
        l1_norm[l1_norm <= gamma]= gamma
        shrink_mat =  1.0 - gamma/l1_norm
        shrink_mat[shrink_mat<=0] = 0.0
        shrink_mat = np.tile(shrink_mat,[1,3])
        shrink_mat = np.reshape(shrink_mat, [n_dipoles, n_coefs*p])
        z *= shrink_mat
    else:
        raise ValueError("n_orient must be 1 or 3")
    # ===== level1, group of coefficients =====
    #y_ind = np.nonzero(active_set_y)[0]
    #array0 = np.arange(0,p).astype(np.int)
    # make this step faster!!   
    if n_orient == 1: 
        #    shrink = np.zeros([n_dipoles,n_coefs])
        #        for i in range(len(y_ind)):
        #            for j in range(n_coefs):
        #                # temporary group in z
        #                z_g = z[i,array0*n_coefs+j]
        #                group_l2_norm = np.sqrt(np.sum(np.real(z_g)**2 + np.imag(z_g)**2))
        #                # if group_l2_norm is 0, then shrink = 0, avoid division by zero
        #                shrink[i,j] =  np.maximum(1.0 - beta/np.maximum(group_l2_norm,beta), 0.0)
        #                z[i,array0*n_coefs+j] *= shrink
        reshape_z = np.reshape(z, [n_dipoles, p, -1])
        group_l2_norm = np.sqrt(np.sum(np.abs(reshape_z)**2, axis = 1))
        group_l2_norm[group_l2_norm <= beta]= beta
        shrink_mat =  1.0 - beta/group_l2_norm
        shrink_mat[shrink_mat<=0] = 0.0
        shrink_mat = np.tile(shrink_mat,[1,p])
        z *= shrink_mat
    elif n_orient == 3:
        #        shrink = np.zeros([len(y_ind)//3, n_coefs])
        #        for i in range(len(y_ind)//3):
        #            for j in range(n_coefs):
        #                # temporary group in z
        #                z_g = np.hstack([z[3*i,array0*n_coefs+j],\
        #                                z[3*i+1,array0*n_coefs+j],\
        #                                z[3*i+2,array0*n_coefs+j]])
        #                group_l2_norm = np.sqrt(np.sum(np.real(z_g)**2 + np.imag(z_g)**2))
        #                # if group_l2_norm is 0, then shrink = 0, avoid division by zero
        #                shrink[i,j] =  np.maximum(1.0 - beta/np.maximum(group_l2_norm,beta), 0.0)
        #                for k in range(3):
        #                    z[3*i+k,array0*n_coefs+j] *= shrink[i,j]
        reshape_z = np.reshape(z, [n_dipoles//3,3, p, -1])
        group_l2_norm = np.sqrt(np.sum(np.sum(np.abs(reshape_z)**2, axis = 2),axis = 1))
        group_l2_norm[group_l2_norm <= beta]= beta
        shrink_mat =  1.0 - beta/group_l2_norm
        shrink_mat[shrink_mat<=0] = 0.0
        shrink_mat = np.tile(shrink_mat,[1,p])
        shrink_mat = np.tile(shrink_mat,[1,3])
        shrink_mat = np.reshape(shrink_mat, [n_dipoles, n_coefs*p])
        z *= shrink_mat
    else:
        raise ValueError("n_orient must be 1 or 3")
    #===== level 2, group of dipoles =====
    for j in range(len(DipoleGroup)):
        # input is a full matrix, so all groups are in it
        z_g = np.ravel(z[DipoleGroup[j],:], order = 'C')
        group_l2_norm = np.sqrt(np.sum(np.real(z_g)**2 + np.imag(z_g)**2))
        alpha_tmp = alpha * DipoleGroupWeight[j]   
        # note that alpha_tmp could be zero!
        if alpha_tmp > 0:
            shrink =  np.maximum(1.0 - alpha_tmp/np.maximum(group_l2_norm,alpha_tmp), 0.0)
            z[DipoleGroup[j],:] *= shrink
      
    active_set_z = np.sum(np.abs(z), axis = 1) > 0
    z = z[active_set_z,:]
    return z, active_set_z
    
    
def tree_obj_hard_coded(y, active_set_y, z, active_set_z, n_coefs, p, 
              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient):
    """ compute the objective function  1/2||y-z||_2^2 + \alpha Omega(z)
    """
    y_ind = np.nonzero(active_set_y)[0]
    z_ind = np.nonzero(active_set_z)[0]
    all_ind = np.union1d(y_ind,z_ind)
    y_large = np.zeros([len(all_ind), n_coefs*p], dtype = np.complex)
    z_large = np.zeros([len(all_ind), n_coefs*p], dtype = np.complex)
    
    y_large[np.searchsorted(all_ind,y_ind),:] = y
    z_large[np.searchsorted(all_ind,z_ind),:] = z
    diff_yz = y_large - z_large
    # expand diff_yz to a full matrix
    n_dipoles = len(active_set_y)
    z_full = np.zeros([n_dipoles, n_coefs*p], dtype = np.complex)
    z_full[active_set_z,:] = z
    tree_norm = get_tree_norm_hard_coded(z_full,n_coefs, p, 
              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient)
    obj_fun = 0.5 * np.sum(np.real(diff_yz)**2 + np.imag(diff_yz)**2) + tree_norm
    return obj_fun
    

def get_tree_norm_hard_coded(z, n_coefs, p, 
              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient):
    """ compute the tree norm  Omega(z)
        z [n_dipoles, n_coefs*p] a full matrix
    """
    tree_norm = 0
    if z.shape[1] != n_coefs*p:
        print "the number of coefficients do not match"
        return 0, 0
    n_dipoles = z.shape[0]
    # ===== level 0 abs =====
    if n_orient == 1:       
        l1_norm = np.abs(z)
        tree_norm += gamma*l1_norm.sum()
    elif n_orient == 3:
        reshape_z = np.reshape(z, [n_dipoles//3,3, -1])
        l1_norm = np.sqrt(np.sum(np.abs(reshape_z)**2, axis = 1))
        tree_norm += gamma*l1_norm.sum()
    else:
        raise ValueError("n_orient must be 1 or 3")
        
    # ===== level 1 group of coefficients =====
    if n_orient == 1:       
        reshape_z = np.reshape(z, [n_dipoles, p, -1])
        group_l2_norm = np.sqrt(np.sum(np.abs(reshape_z)**2, axis = 1))
        tree_norm += beta*group_l2_norm.sum()
    elif n_orient == 3:
        reshape_z = np.reshape(z, [n_dipoles//3,3, p, -1])
        group_l2_norm = np.sqrt(np.sum(np.sum(np.abs(reshape_z)**2, axis = 2),axis = 1))
        tree_norm += beta*group_l2_norm.sum()
    else:
        raise ValueError("n_orient must be 1 or 3")
    #===== level 2, group of dipoles =====
    for j in range(len(DipoleGroup)):        
        alpha_tmp = alpha * DipoleGroupWeight[j] 
        z_g = np.ravel(z[DipoleGroup[j],:], order = 'C')
        group_l2_norm = alpha_tmp*np.sqrt(np.sum(np.real(z_g)**2 + np.imag(z_g)**2))
        tree_norm += group_l2_norm
        
    return tree_norm
    

    
### test 
#n_dipoles = 6
#n_coefs = 5
#p = 2
#y = np.random.randn(n_dipoles, n_coefs*p) + 1j * np.random.randn(n_dipoles, n_coefs*p)
#alpha, beta, gamma = 1,1,1
#DipoleGroup = list([np.array([0,1,2]), np.array([3,4,5])]) 
#DipoleGroupWeight = np.ones(2)/2
#
#n_orient = 3
#active_set_y = np.ones(n_dipoles, dtype = np.bool)
#z, active_set_z  = prox_tree_hard_coded_full_matrix(y, n_coefs, p, 
#              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient)      
#obj1 = tree_obj_hard_coded(y, active_set_y, z, active_set_z, n_coefs, p, 
#              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient)
#
#obj2 = tree_obj_hard_coded(y, active_set_y, z-np.random.randn(z.shape[0],z.shape[1])*1E-2,
#                         active_set_z, n_coefs, p, 
#              alpha, beta, gamma, DipoleGroup, DipoleGroupWeight, n_orient)              
#    
#print obj2-obj1    
#    
#        
        
          
