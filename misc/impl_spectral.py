#!/usr/bin/env python

'''
Spectral implementation.
Graph cut point of view of clustering. Dimensionality reduction, and then clustering by K-means on the dim-reduced data.

--Cover:
Spectral (RatioCut), make use of `sklearn.cluster.spectral_clustering`.

--Math:
Given graph $G=(V,E)$, find balanced subgraphs $A_1,\dots,A_k$, such that vertices in $A_i$ are tightly connected with each other, and loosely connected between subgraphs. That is, solve the mincut problem,
$$\min_{A_1,\dots,A_k}RatioCut(A_1,\dots,A_k)=\sum_i[cut(A_i, \bar A_i)/|A_i|]$$
, $|A_i|$ is the size, we also have $vol(A_i)$ meaning the sum of degree, and 
$$cut(A,\bar A)=\sum_{i\in A, j\in \bar A}w_{ij}$$

To solve this problem, we denote vertices in $A_i$ as a vector
$$h_i=\begin{cases}
1/\sqrt{|A_i|}, if v_j \in A_i\\
0, otherwise
\end{cases}$$
, so we have $k$ vectors, each of which is piecewise constant.

Define $W,D,L$ as adjacency matrix, degree matrix and Laplacian matrix of graph $G$,
and $D=diag(sum(W, axis=1))$, $L=D-W$.

We note that,
$$cut(A,\bar A)=h^TLh$$
, and $h_i\perp h_j$, $h^Th=|A|$. And now the problem is to solve
$\min\sum_ih_i^TLh_i$, s.t. $h_i\perp h_j$, $h_i^Th_i=|A_i|$,
and this is amount to
$\min Tr(H^TLH)$, s.t. $H^TH=I$.

By Lagrange multiplier, we know that the solution is the smallest $k$ eigenvectors of $L$.

--Practical Issue
1) Typically, we choose $k$ such that the first $k$ eigenvalues vary slowly, where there is a big gap between $k$ and $k+1$.

2) On high-dim data, spectral is faster than K-means, since the DR effect.
'''

import numpy as np
from scipy import spatial
from sklearn import cluster
import matplotlib.pyplot as plt
import time

def radial_kernel(X, sigma=10.0):
    '''
    Compute the similarity matrix, using radial kernel.
    `scipy.spatial.distance.pdist` doesn't return a matrix, but it's fast.
    
    The following implementation is much faster than plain `for`-loops,
    and it's comparable with `distance.pdist` when sample size is small,
    and this is the fastest version I can do.
    '''
    # cosine similarity
    C = np.dot(X, X.T)
    # magnitude
    M = np.diag(C).reshape((X.shape[0],1))
    D = (M - 2 * C).T + M
    S = np.exp(-D/sigma)
    return S
    
def radial_kernel2(X, sigma=10.0):
    '''
    Use `scipy.spatial.distance`, `pdist` and `squareform`.
    '''
    D = spatial.distance.pdist(X)**2
    D = spatial.distance.squareform(D)
    D = np.exp(-D/sigma)
    return D

def spectral(W, k):
    '''
    Parameters:
    W: adjacency matrix,
    k: number of clusters.
    Returns:
    label: 1-d array, cluster labels.

    Not as fast as `sklearn.cluster.spectral_clustering`.
    '''
    n = W.shape[0]
 
    # degree matrix
    D = np.diag(W.sum(axis=1))
    # Laplacian matrix
    L = D - W
    # L is positive semi-definite, svd == eig
    # it takes up the bulk of time cost, maybe some powerful techniques used 
    # (for sparse matrix?) in `sklearn.cluster.spectral_clustering`.
    U = np.linalg.svd(L)[0][:,-k:]

    # cluster with k-means
    ret = cluster.k_means(U, k)
    return ret[1]

def adjacency_matrix(X, k):
    '''
    Parameters:
    X: data with shape (n_sample, m_dim),
    k: k nearest neighbors to form similarity graph.
    Returns:
    W: adjacency weight matrix.
    '''
    n = X.shape[0]
    S = radial_kernel(X)
    kd = spatial.KDTree(X)
    # kNN matrix
    J = kd.query(X, k+1)[1]  # +1 for taking self as neighbor
    W = np.zeros(S.shape)
    for i in range(n):
        W[i, J[i, 1:]] = 1
    # mutual kNN matrix
    W *= W.T
    # this is the final adjacency weight matrix
    W *= S
    return W

if __name__ == '__main__':
    #rf = lambda r,t: (r*np.cos(t), r*np.sin(t))
    #t = np.linspace(np.pi/3, np.pi/2, num=20)
    #r = np.linspace(2, 0, num=20)    
    #X = np.array(map(rf, r, t))
    #t = np.linspace(np.pi/3, -np.pi/6, num=20)
    #X = np.vstack((X, np.array(map(rf, r, t))+(1.0,1.0)))
    #X = np.vstack((X, np.random.sample(size=(10,2))))
    X = np.random.sample(size=(1000,2))

    W = adjacency_matrix(X, k=10)
    
    n_cluster = 3
    L = spectral(W, k=n_cluster)
    L2 = cluster.spectral_clustering(W, n_clusters=n_cluster)
    L3 = cluster.k_means(X, n_clusters=n_cluster)[1]
    
    color = 'rbcmyg'
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for i in range(n_cluster):
        ax.scatter(X[L==i, 0], X[L==i, 1], color=color[i])
    for i in range(n_cluster):
        ax2.scatter(X[L2==i, 0], X[L2==i, 1], color=color[i])
    for i in range(n_cluster):
        ax3.scatter(X[L3==i, 0], X[L3==i, 1], color=color[i])
    plt.show()
