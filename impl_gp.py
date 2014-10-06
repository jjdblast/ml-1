#!/usr/bin/env python

''' Gaussian Process.

Try to implement GP predictive distribution by noise-free and noisy observations.
1) With noise-free training data, y = f.
    The only source of randomness is that of the uncertainty of f, which is 
    represented in the GP prior: f ~ NP(m, K).
    And since there is no other randomness of f, so the observed values are
    taken to be the true f, no other possible values.
    So, given the observed f, what about the unseen f'? Maginalization!
    Yes, that is, we have (f,f') ~ NP(m, C) -> (f'|f) ~ (m', C').
2) With noisy training data, y = f + e.

Actually, noise-free and nosiy predictive distribution, they share the same
predictive formula, except that, the covariance matrices differ by $\sigma^2I$
for noisy one.
'''

from numpy import *
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

kernels = (lambda x1,x2: exp(-dot((x1-x2),(x1-x2))/2),
           lambda x1,x2: exp(-abs(x1-x2)),
)

means = (lambda x: 0,
         lambda x: x,
)

def mvn_sample(mean, cov, U=None):
    if U is None:
        U = asarray([random.normal() for i in range(mean.size)])
    L = linalg.cholesky(cov)
    return mean + dot(L,U), U

def get_posterior_f(x,y,sigma, mean,cov):
    K = cov + sigma**2*diag(ones(x.size))
    K = linalg.inv(K)
    M = dot(K,dot(K,y))
    return M, K

def sample_and_plot_f(x, M, K, nsample=3, ax=None):
    for i in range(nsample):
        y,U = mvn_sample(M, K+float('1e-8')*diag(ones(x.size)))
        if ax is None:
            plt.plot(x,y)
            continue
        ax.plot(x,y)

def noisy_predictive(mean_func, cov_func, noise=float('1e-8')):
    # observed data, compute the covariance K1.
    x1 = asarray([-2.7,-0.9,0.3,1.1,1.9])
    f1 = random.sample(5)

    M1 = asarray([mean_func(i) for i in x1])
    K1 = asarray([cov_func(i,j) for i in x1 for j in x1]).reshape((x1.size,x1.size))

    # data to predict,
    x2 = linspace(-3,3,100)
    M2 = asarray([mean_func(i) for i in x2])
    K2 = asarray([cov_func(i,j) for i in x2 for j in x2]).reshape((x2.size,x2.size))
    K21 = asarray([cov_func(i,j) for i in x2 for j in x1]).reshape((x2.size,x1.size))
    
    # compute the new mean and covariance, M_predictive and K_predictive.
    # this is for predictive distribution, conditioning on the observed (x1,f1).
    K1_inv = linalg.inv(K1+noise*diag(ones(x1.size)))
    M_predictive = dot(K21,dot(K1_inv,f1))
    K_predictive = K2 - dot(K21,dot(K1_inv,K21.T))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.scatter(x1,f1,marker='+')
    sample_and_plot_f(x2,M2,K2,5, ax=ax1)
    sample_and_plot_f(x2,M_predictive,K_predictive,5, ax=ax2)
    plt.show()


if __name__ == '__main__':
    mean = means[0]
    kernel = kernels[0]
    noisy_predictive(mean, kernel, float('1e-10'))
