#!/usr/bin/env python

from numpy import *
from scipy import stats
import matplotlib.pyplot as plt
import o_stat

target = lambda x: .5*x

m = 7
n = 2
data = o_stat.simulated_sample(target, size=m, alpha=0)
X = asarray((ones(m), data[:,0])).T
Y = data[:,1]
T = asarray([(i,j) for i in linspace(-2,2,50) 
                    for j in linspace(-2,2,50)])

t_cov= diag(ones(n)) * .5
normal_pr = o_stat.mv_normal(zeros(n), t_cov)
prior = asarray(map(normal_pr, T))

d_cov = diag(ones(m)) * .5
normal_lk = (o_stat.mv_normal(dot(X,_), d_cov) for _ in T)
likelihood = asarray(map(lambda f:f(Y), normal_lk))

A = 2*dot(X.T,X) + 2*diag(ones(2))
mean = 2*dot(linalg.inv(A), dot(X.T, Y))
p_cov = linalg.inv(A)
normal_ps = o_stat.mv_normal(mean, p_cov)
posterior = asarray(map(normal_ps, T))

fig = plt.figure()
ax_pr = fig.add_subplot(131)
ax_pr.imshow(prior.reshape(50,50), extent=[-2,2,-2,2])

ax_lk = fig.add_subplot(132)
ax_lk.imshow(likelihood.reshape(50,50), extent=[-2,2,-2,2])

ax_ps = fig.add_subplot(133)
ax_ps.imshow(posterior.reshape(50,50), extent=[-2,2,-2,2])

plt.show()
