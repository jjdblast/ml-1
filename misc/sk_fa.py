#!/usr/bin/env python


'''
Factor Analysis.

This is not an implementation from scatch, just a trial with scikit-learn.
Run a comparison between FA & PCA using the same amount of components/factors.

--Lesson
1) sklearn.fa takes too much time, I can't take this. After searching the web, maybe the EM tends to converge slowly in FA, and some other more efficient methods are proposed, such as ECME.

'''

import sklearn.decomposition as dcmp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


if __name__ == '__main__':
    X = ndimage.imread('/home/alan/Pictures/lena.png', flatten=True)
    
    fa = dcmp.FactorAnalysis(n_components=100)
    fa.fit(X)
    X_new_fa = fa.transform(X)
    
    pca = dcmp.PCA(n_components=100)
    pca.fit(X)
    X_new_pca = pca.transform(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(np.dot(X_new_fa, fa.components_))
    ax2.imshow(np.dot(X_new_pca, pca.components_))
    fig.savefig('pic/fa.png')
