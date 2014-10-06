#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
LR Implementation.
Before implementing, do the math carefully.

--Lesson
1) I found that without regularization, BFGS encountered overflow in `exp` of `sig`,
that's because `w` grows large.
2) One note about impl: always use 1-d array to represent vector. Since some methods in
numpy/scipy does so.
3) Over some randomly generated data set, it seems BFGS failed to converge...

--Gradient method
$L=\prod p^y(1-p)^{1-y}$ -> $L=\sum y\log p+(1-y)\log(1-p)$,
$g=\sum[(y/p+(y-1)/(1-p))\nabla_wp]$,
$p=1/(1+e^{-w^Tx})$ -> $\nabla_wp=e^{-w^Tx}x/(1+e^{-w^Tx})^2=p(1-p)x$,
$g=\sum[(y/p+(y-1)/(1-p)) * p(1-p)x]=\sum[y(1-p)x+(y-1)px]=\sum[yx-px]$.
Here $L$ is likelihood, we try to maximize it, so we should use gradient ascent, right? 
If $L$ is deviance, use gradient descent.

--Newton's method
$h=\nabla_wg=\sum[-x\nabla_wp]=\sum[-p(1-p)xx^T]$.
'''

from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op

sig = lambda X,w: 1.0/(1+exp(-dot(X, w)))
sig_f1 = lambda X,Y,w: dot(X.T, Y - sig(X, w))
sig_f2 = lambda X,Y,w: dot(sig(X, w)*(sig(X, w)-1)*X.T, X)

def ga(X, Y, w, l=0.01, eps=1e-5):
    '''
    Gradient ascent, for mle of lr.
    '''
    g = ones(w.shape)
    while linalg.norm(g) > eps:
        g = sig_f1(X, Y, w)
        w += l*g
    return w

def nt(X, Y, w, l=0.01, eps=1e-5):
    '''
    Newton's method in optimization, for mle of lr.
    '''
    g = ones(w.shape)
    while linalg.norm(g) > eps:
        g = sig_f1(X, Y, w)
        h = sig_f2(X, Y, w)
        w -= l*dot(linalg.inv(h), g)
    return w

def quasi_nt(X, Y, w, method='BFGS'):
    '''
    Use quasi-Newton (BFGS) from `scipy.optimize`.
    Deviance:
    $L=-2(\sum y\log p+(1-y)\sum\log(1-p))$.
    '''

    def obj(w, X, Y):
        P = sig(X, w)
        P = vstack((P, 1-P)).T
        Y = vstack((Y, 1-Y)).T
        Z = P * Y
        return -Z.sum() + 0.1*linalg.norm(w)

    # Without specifying jac, bfgs didn't give a good w.
    def jac(w, X, Y):
        g = sig_f1(X, Y, w)
        return g + 0.2*w

    w = op.minimize(obj, w, args=(X, Y), jac=jac, method=method)
    return w.x

def plot_class2(X, Y, w):
    '''
    Boundary: $x_2=-(w_0+w_1x_1)/w_2$.
    '''
    plt.scatter(X[Y==0,1], X[Y==0,2], marker='*', color='b')
    plt.scatter(X[Y==1,1], X[Y==1,2], marker='^', color='r')

    def curve(x2, x1):
        x = ones(w.shape)
        for i in range(1, x.size, 2):
            x[i] = x1**((i+1)/2)
        for i in range(2, x.size, 2):
            x[i] = x2**((i+1)/2)
        return dot(w, x)
    
    x1 = linspace(X[:,1].min(), X[:,1].max(), num=30)
    x2 = [op.root(curve, 0, i).x for i in x1]
    plt.plot(x1, x2, 'b-')
    plt.show()


if __name__ == '__main__':
    X = random.normal(size=(170,2))
    X[:90,] += 2
    Y = ones(170)
    Y[:90,] = 0

    X = hstack((X, X**2))
    X = hstack((ones((X.shape[0],1)), X))
    w = zeros(X.shape[1])
    w1 = nt(X, Y, w)
    print w1
    plot_class2(X, Y, w1)
