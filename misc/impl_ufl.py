#!/usr/bin/env python

'''
Unsupervised feature learning.
Good features are one of the most important things (if not the most).

--Cover
Dictionary learning by K-means (standard/sperical).

--Lesson
1) Tried LR/SVC/RFC on `sklearn.datatsets.load_digits' dataset.
Without data whitening, LR perfroms best on raw pixel input, SVC perform worst.
With whitening, SVC performs best.

2) More clusters learned, better LR performs.
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.datasets import load_digits
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import svm
import o_stat

def zca(X, eps=1e-9):
    # centering
    X -= X.mean(axis=0)
    # covariance
    C = np.cov(X, rowvar=0)
    U,D,V = np.linalg.svd(C)
    # decorrelating
    Z = np.dot(X, U)
    # whitening
    Z = np.dot(Z, np.diag(1.0/np.sqrt(D+eps)))
    # rotating back
    Z = np.dot(Z, U.T)
    return Z

def test_zca():
    np.random.seed(0)
    X = np.vstack((np.random.sample(size=200), np.random.normal(size=200))).T
    Z = zca(X)

    print np.cov(X, rowvar=0)
    print np.cov(Z, rowvar=0)
    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1])
    plt.subplot(1,2,2)
    plt.scatter(Z[:,0], Z[:,1])
    plt.show()

def spkmeans(X):
    pass

def digit_recognize():
    X = load_digits()
    y = X.target
    X = X.data

    n,p = X.shape
    mask = np.zeros(n, dtype=bool)
    train = np.random.choice(n, 3*n/5, replace=False)
    mask[train] = True

#    X = zca(X)
    w = np.sqrt(p)
    X = fe(X.reshape((-1,w,w)), 4)

    lr = lm.LogisticRegression(penalty='l2', C=1.0)
    svc = svm.SVC(kernel='rbf', C=3)
    rfc = RFC()

    lr.fit(X[mask], y[mask])
    svc.fit(X[mask], y[mask])
    rfc.fit(X[mask], y[mask])

    pred = [_.predict(X[~mask]) for _ in (lr, svc, rfc)]
    r = np.array([np.sum(p == y[~mask]) for p in pred])
    q = np.sum(~mask)
    print r/float(q)


if __name__ == '__main__':
    for i in range(10):
        digit_recognize()
