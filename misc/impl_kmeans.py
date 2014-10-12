#!/usr/bin/env python

'''
K-means implementation.
Pretty simple and also popular, useful for compactness clustering..

--Math
1) What?
Partition the data into $K$ groups such that an objective function is optimized,
$$\min_{C1,\ldots,C_K}\sum_k\frac{1}{|C_k|}\sum_{i,j\in C_k}(x_i-x_j)^2$$.
The objective function here is the total within-class variation, and the common choice is pairwise-point-distance.
It can be shown,
$$\frac{1}{|C_k|}\sum_{i,j}(x_i-x_j)^2 = 2\sum_{i}(x_i-\bar x)^2$$
, that is twice the variance of class $k$.

2) How?
Iteratively reassign each point to the closest centroid in each iteration.
This guarantees that the objective function decreases, but it just provides a local optimum.
Run it several times with different initial states.

--Lesson
1) The initial step, randomly assign a cluster to each point, might cause a situation in which one or more clusters end up empty. So randomly pick K points from the dataset to kick off the algorithm.
Seems like more robust than randomly seeding.

2) K-means++ is an algorithm to carefully select seeds, which makes K-means converge faster.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster

def kmeans(X, k, tol=1e-8, max_iter=100, verbose=True):
    '''
    Parameters:
    X: array-like with shape (nsamples, nfeatures), data to be clusterd.
    k: integer, number of clusters.
    tol: float, tolerance to stop.

    Return:
    C: final centers
    I: final labels
    D: final total variation
    '''
    C = X[np.random.choice(X.shape[0], size=k, replace=False)]
    I = None  # not necessary, module/class/function scope in python, not like C++
    tot = 0.0
    for it in range(max_iter):
        D = np.asarray([np.sum((X - i)**2, axis=1) for i in C]).T
        I = D.argmin(axis=1)
        _tot = np.sum([np.sum(X[I==i].var(axis=0)) for i in range(k)])
        if np.abs(tot - _tot) <= tol:
            break
        C = np.asarray([X[I==i].mean(axis=0) for i in range(k)])
        tot = _tot
        if verbose:
            print 'iter: {0}, tot: {1}'.format(it, tot)
    return C, I, D

def spkmeans(X, k, tol=1e-8, max_iter=100, verbose=True):
    '''Naive Spherical Kmeans.'''
    # scale X to unit length, practice with einsum
    L = np.einsum('ij,ij->i', X, X)  # ->i, sum over each i, equiv to axis=1
    L = np.sqrt(L)
    X = np.einsum('i...,i...->i...', X, 1.0/L)  # divide by length

    n, p = X.shape
    # randomly pick k centers
    _I = np.random.choice(n, k, replace=False)
    C = X[_I]
    delta = 1.0
    tot = 0.0
    iterr = 0
    while delta > tol:
        # similarity matrix
        D = 1 - np.dot(X, C.T)
        # new label
        I = D.argmin(axis=1)
        # new delta
        _tot = D[np.arange(I.size), I].sum()
        delta = np.abs(tot - _tot)
        tot = _tot
        # new centers, deal with missing clusters
        C = np.asarray([X[I==i].mean(axis=0) for i in range(k)])
        mask = np.all(np.isnan(C), axis=1)
        n_mask = mask.sum()
        if n_mask > 0:
            C[mask] = X[np.random.choice(n, n_mask, replace=False)]
        iterr += 1
        if verbose:
            print 'iter: {0}, tot: {1}'.format(iterr, tot)
        if iterr > max_iter:
            break
    return C, I, D


if __name__ == '__main__':
    X = np.asarray(ndimage.imread('pic/Penguin.jpg', flatten=True))
    Y = np.zeros(shape=X.shape)
    k = 100
    C,I,D = kmeans(X, k)
    for i in range(k):
        Y[I==i] = C[i]
    Y = Y.reshape(X.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(X, cmap='gray')
    ax2.imshow(Y, cmap='gray')
    plt.show()
