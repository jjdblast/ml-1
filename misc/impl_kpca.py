#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PCA implementation.
PCA serves mainly as a dimensionality reduction technique.

Seriously why do we use PCA?
Just tranform the data to obtain uncorrelated features, or
to represent the data using fewer features?

--Cover:
max-variance interpretation (SVD/Eigen), conventional PCA.
Kernel PCA.

--TODO:
sparse PCA, haven't understand the idea yet, -_-//.

--Math:
1) Max-variance
Find $w$ which solves
\begin{equation*}
\begin{aligned}
&\max_w & & \|Xw\|^2\\
&\text{s.t.} & & \|w\|=1
\end{aligned}
\end{equation*}
Solve it via Lagrange multiplier, resulting into Eigen-decomposition.
Or via SVD, which is more convinient solution to come up with.

2) Min-reconstruction error
'''


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import spatial


#---------------------------------------------------------
# Conventional PCA
# Code is obsolete, too easy, no need to write these functions.

def get_pc(X, eps=1e-8):
    '''
    Ignore those pc with $0$-singular-value, 
    since they don't explain any variance.
    '''
    U,S,V = np.linalg.svd(X)
    return V[np.abs(S)>eps]
    
def get_score(X, C):
    return np.dot(X, C.T)

def reconstruct(M, C, k=3):
    '''
    Reconstruct original data.
    `M': score matrix,
    `C': pc matrix,
    `k': how many pc to use.
    '''
    return np.dot(M[:,:k], C[:k,:])

#--------------------------------------------------------
# Kernel PCA

def kernel_radial(X, c=10):
    # compute pairwise distance
    D = spatial.distance.pdist(X, metric='euclidean')
    D = spatial.distance.squareform(D)**2

    # kernel matrix 
    K = np.exp(-D/c)
    return K

def kernel_poly(X, c=0, p=2):
    D = np.dot(X, X.T)
    C = c * np.outer(np.ones(X.shape[0]), np.ones(X.shape[0]))
    K = D + C
    K = K**p
    return K

def test_kpca():
    import o_stat
    X = o_stat.circle_sample([1.5,5.0,10.0,3.5], ndim=2)
    n = X.shape[0]
    m = 4

    # no centering
    K = kernel_poly(X, 10, 5)
    USV = np.linalg.svd(K)
    Z = USV[0]*np.sqrt(USV[1])
    
    # centering
    M = np.eye(N=n) - np.outer(np.ones(n), np.ones(n))
    K2 = np.dot(np.dot(M, K), M)
    USV2 = np.linalg.svd(K2)
    Z2 = USV2[0]*np.sqrt(USV2[1])

    color = 'bcrgm'
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for i in range(m):
        f = i*n/m
        t = (i+1)*n/m
        ax1.scatter(X[f:t,0], X[f:t,1], color=color[i])
        ax2.scatter(Z[f:t,0], Z[f:t,1], color=color[i])
        ax3.scatter(Z2[f:t,0], Z2[f:t,1], color=color[i])
    plt.show()

def test_kpca2():
    import o_stat
    X = np.vstack((np.random.sample(size=(20,2))+(-1.0,1.0),
                    np.random.sample(size=(30,2))+(1.0,1.0),
                    np.random.sample(size=(15,2))+(0, -1.0)))
    K = kernel_poly(X, 1)
    U,S,V = np.linalg.svd(K)
    Z = U*np.sqrt(S)

    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1])
    plt.subplot(1,2,2)
    plt.scatter(Z[:,0], Z[:,1])
    plt.show()

if __name__ == '__main__':
    test_kpca()
