#!/usr/bin/env python

''' Regularization. 
It seems that both `numpy' and `scipy' contain stat util functions,
which look quite similar, here is the difference:
1) Random sampling functions are from `numpy.random', such as:
`random' - uniformly sampling in [0.0, 1.0),
`beta' - Beta sampling in [0.0, 1.0),
`normal' - Gaussian smapling,
...

2) Distributions are from `scipy.stats', such as:
`beta' - `beta(10,10).pdf(.5)' pdf,
`norm' - `norm(1,10).pdf(1)' pdf,
...
They are `shape parameters'.

'''

from numpy import *
from scipy import stats
from matplotlib import pyplot as plt
from o_stat import simulated_sample


target = lambda x: sin(2*pi*x)

def poly_feature(X, d):
    X = asarray([X**i for i in range(d+1)]).T
    return X

def poly_model(X, Y, d, r=0):
    X = poly_feature(X, d)
    w = ones(X.shape[1])
    w[0] = 0
    b = dot(linalg.inv(dot(X.T,X)+r*diag(w)), dot(X.T,Y))
    return b

if __name__ == '__main__':
    data = simulated_sample(f=target, size=99)
    model_1_b = poly_model(data[:,0], data[:,1], 1)
    model_3_b = poly_model(data[:,0], data[:,1], 3)
    model_7_b = poly_model(data[:,0], data[:,1], 7)
    model_15_b = poly_model(data[:,0], data[:,1], 15)

    model_1_b_ = poly_model(data[:,0], data[:,1], 1, .3)
    model_3_b_ = poly_model(data[:,0], data[:,1], 3, .3)
    model_7_b_ = poly_model(data[:,0], data[:,1], 7, .3)
    model_15_b_ = poly_model(data[:,0], data[:,1], 15, .3)

    model_1_b__ = poly_model(data[:,0], data[:,1], 1, 1.9)
    model_3_b__ = poly_model(data[:,0], data[:,1], 3, 1.9)
    model_7_b__ = poly_model(data[:,0], data[:,1], 7, 1.9)
    model_15_b__ = poly_model(data[:,0], data[:,1], 15, 1.9)

    fig = plt.figure()
    X = sort(data[:,0])

    ax = fig.add_subplot(221)
    ax.scatter(data[:,0], data[:,1], c='red')
    ax.plot(X, dot(poly_feature(X,1),model_1_b), c='green', ls='-', lw=2)
    ax.plot(X, dot(poly_feature(X,1),model_1_b_), c='green', ls='--', lw=2)
    ax.plot(X, dot(poly_feature(X,1),model_1_b__), c='green', ls=':', lw=2)

    ax = fig.add_subplot(222)
    ax.scatter(data[:,0], data[:,1], c='red')
    ax.plot(X, dot(poly_feature(X,3),model_3_b), c='black', ls='-', lw=2)
    ax.plot(X, dot(poly_feature(X,3),model_3_b_), c='black', ls='--', lw=2)
    ax.plot(X, dot(poly_feature(X,3),model_3_b__), c='black', ls=':', lw=2)

    ax = fig.add_subplot(223)
    ax.scatter(data[:,0], data[:,1], c='red')
    ax.plot(X, dot(poly_feature(X,7),model_7_b), c='grey', ls='-', lw=2)
    ax.plot(X, dot(poly_feature(X,7),model_7_b_), c='grey', ls='--', lw=2)
    ax.plot(X, dot(poly_feature(X,7),model_7_b__), c='grey', ls=':', lw=2)

    ax = fig.add_subplot(224)
    ax.scatter(data[:,0], data[:,1], c='red')
    ax.plot(X, dot(poly_feature(X,15),model_15_b), c='purple', ls='-', lw=2)
    ax.plot(X, dot(poly_feature(X,15),model_15_b_), c='purple', ls='--', lw=2)
    ax.plot(X, dot(poly_feature(X,15),model_15_b__), c='purple', ls=':', lw=2)

    fig.canvas.draw()
    fig.savefig('pic/regularize.png')
