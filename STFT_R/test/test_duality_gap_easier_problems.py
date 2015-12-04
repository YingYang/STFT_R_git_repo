# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:18:11 2014

@author: ying
"""

# a simple lasso problem
import numpy as np


# structure, b groups
# group is in the tree order

def proximal(b, Lambda, groups):    
    for i in range(len(groups)):
        group_ind = groups[i]
        norm_b  = np.sqrt(np.sum((b[group_ind])**2))
        if norm_b < Lambda[i]:
            norm_b = Lambda[i]
        shrink =  1.0 - Lambda[i]/norm_b
        b[group_ind] *= shrink
    return b 


# solve the lasso problem
def solve_tree_lasso_ista(Y,X,Lambda,  groups,tol = 1e-5, maxit = 100):
    n,p = X.shape
    b = np.zeros(p)
    for i in range(maxit):
        residual = Y-X.dot(b)
        gradient = -X.T.dot(residual)
        tmp_Lambda = Lambda/Lipschitz_const
        b_new = proximal(b-gradient/Lipschitz_const, tmp_Lambda, groups)
        diff = np.sum(np.abs(b_new-b))
        b = b_new
        print "iteration %d" %i
        print "diff = %e" % diff
        #print "b * gradient %e" % b.dot(gradient)
        obj = 0.5*np.sum(residual**2) 
        for j in range(len(groups)):
            group_ind = groups[j]
            norm_b_group  = np.sqrt(np.sum((b[group_ind])**2))
            obj += norm_b_group * Lambda[j]
        if diff <  tol:
            break
        print "obj = % e"%obj
    #print "b * gradient %e" % b.dot(gradient)
    return b, obj
       
       
def compute_dual_gap(Y,X,Lambda, groups, b, obj, maxit = 100, tol = 1e-2):
    z = X.dot(b)
    u = z-Y
    dual_obj = 0.5*np.sum(u**2)  - np.dot(u,z) 
    dual_gap = obj-dual_obj
    gradient = X.T.dot(u)
    active_set = nonzero(np.abs(b))[0]
    group_active_set = np.zeros(len(groups))
    a = gradient.copy()
    for i in range(len(groups)):
        if np.intersect1d(groups[i],active_set).size>0:
            group_active_set[i] = True
            grad_group = b[groups[i]]/np.sqrt(np.sum(np.abs(b[groups[i]])**2))
            a[groups[i]] += Lambda[i]*grad_group

    # minimize gradient + lamgda_g + vg
    vg = np.zeros([b.shape[0],len(groups)])
    groups_non_active = [groups[k] for k in range(len(groups)) 
                                  if group_active_set[k] == False]
    Lambda_non_active = Lambda[group_active_set == False]
    for j in range(Maxit):
        old_obj = obj
        for i in range(len(groups_non_active)):
            tmp = a.copy()
            for k in range(len(groups_non_active)):
                if k!= i:
                    tmp_vg = vg[:,k]
                    tmp[groups_non_active[k]] += tmp_vg[groups_non_active[k],:]
            candidate = -tmp
            group_norm = np.sqrt(np.sum((np.abs(candidate[groups_non_active[i]]))**2))
            tmp_vg = np.zeros(len(b))
            if group_norm > Lambda_non_active[i]:
                tmp_vg[groups_non_active[i]] = candidate[groups_non_active[i]] * Lambda_non_active[i]/group_norm            
            else:
                tmp_vg[groups_non_active[i]] = candidate[groups_non_active[i]]
            
            vg[:,i] = tmp_vg
            residual = tmp+vg[:,i]
            old_obj = obj
            obj = np.sum(residual**2)
            print i
            print "obj diff % f" % (old_obj - obj)
            print "obj %f" %obj

        if old_obj - obj< tol:
            break
    feasible_dist = np.sqrt(obj)
    if obj >= tol:
        dual_gap = np.inf
    return dual_gap, feasible_dist
        

    

n = 100
p = 20
b = np.zeros(p)
b[0:3] = 1.0

groups = [np.arange(0,5), np.arange(5,10),
          np.arange(10,15), np.arange(15,20),
          np.arange(0,10), np.arange(10,20)]
Lambda = np.array([5,5,5,5,10,10])*5


#groups = list()
#for i in range(20):
#    groups.append(np.array([i]))
#Lambda = np.ones(20)*10

X = np.random.randn(n,p)
Y = np.random.randn(n)*2 + X.dot(b)

Lipschitz_const = np.linalg.eig(X.T.dot(X))[0][0]
b_hat, obj = solve_tree_lasso_ista(Y,X,Lambda,  groups ) 
print b_hat,obj

dual_gap, feasible_dist = compute_dual_gap(Y,X,Lambda, groups, b_hat, obj, maxit = 100, tol = 1e-2)



# compute the dual gap
active_groups = groups[]
b_hat1,obj1 = solve_tree_ista(Y,X[:,current_active_set],Lambda) 
b_hat2 = np.zeros(p)
b_hat2[current_active_set] = b_hat1
gradient = -X.T.dot(Y-X.dot(b_hat2))


dual_gap = obj1 - dual_obj
print dual_gap

Au = X.T.dot(u)
print np.sum(np.abs(Au - gradient))  # Au = gradient

# a simple feasibility check for group lasso
##  Au + \sum b_g = 0
print "The dual solution is feasible?"
print np.max(np.abs(Au)) <= Lambda
    