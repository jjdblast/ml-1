#!/usr/bin/env python
''' Wheels of stat. '''

from numpy import *
from bisect import bisect_left
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def sample(x, size=1, prob=None):
    if prob is not None:
        cum = cumsum(prob)
    else:
        cum = cumsum(ones(x.size))
    bins = cum/float(cum[-1])
    sampled = random.random(size)
    slot = array(map(lambda x:bisect_left(bins,x), sampled))
    return x[slot]

def simulated_sample(f, size=8, rng=array([0,1]), alpha=0.2):
    x = rng[0] + random.random(size)*(rng[1] - rng[0])
    noise = random.normal(size=size) * alpha
    y = f(x) + noise
    return asarray((x,y)).T

def mv_normal(mean, cov):
    n = mean.size
    a = 1.0/(power(2*pi, .5*n)*power(linalg.det(cov),.5))
    b = linalg.inv(cov)
    f = lambda x: a*exp(-.5*dot(dot(x-mean,b),x-mean))
    f.mean = mean
    f.cov = cov
    return f

def circle_sample(radius, ndim=2):
    theta = linspace(0, pi*2, 90)

    X = []
    for r in radius:
        if ndim==2:
            X.extend([(r*cos(t), r*sin(t)) for t in theta])
        elif ndim==3:
            X.extend([(r*cos(t), r*sin(t), r*sin(t)) for t in theta])
    X = asarray(X)
    X += random.sample(size=X.shape)*.5 - 0.25
    return X

def show_images(X, h, w, col):
    n,p = X.shape
    if w*h != p:
        return -1
    X.shape = (-1, w, h)
    Y = vstack([hstack(X[i*col:(i+1)*col]) for i in range(n/col)])
    plt.imshow(Y, cmap='gray')
    plt.show()
    
def whiten(X, eps=1e-5):
    ''' ZCA. '''
    # centering
    X = X - X.mean(axis=0)
    # covariance
    C = dot(X.T, X)
    # Eigen or SVD
    U,D,V = linalg.svd(C)
    # decorrelate
    X = dot(X, V.T)
    # whitenning
    X = dot(X, diag(1.0/sqrt(D+eps)))
    # rotate back
    X = dot(X, V)
    return X


if __name__ == '__main__':
    X = random.sample(size=(100,3))
    Z = whiten(X,1)

