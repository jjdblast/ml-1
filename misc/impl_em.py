#/usr/bin/env python
# -*- coding: utf-8 -*-

'''
EM implementation for GMM.

Initially, I intended to do it for FA, however I had a hard time deriving the covariance matrix of uniqueness $\epsilon$. In M-step, the partial derivative w.r.t. $\Phi$ is computed, by setting it to 0, I could have the solution $\Phi$, but since $cov(\epsilon)$ is diagonal, so only the diagonal entries taken?

A coordinate ascent algorithm for MLE/MAP parameter estimation in latent variable models.
In MLE, we estimate the parameters by maximizing the log-likelihood. In some situations, it's hard to maximize the log-likelihood via the usual methods (such as gradient, newton). The basic idea of EM is that, it maximize a lower-bound of the log-likelihood, which is more tractable. 
We have proof that maximize the lower bound a coordinate ascent way is guaranteed to improve the lower bound.

--Math
KEYWORD: Jensen's inequality.

1) carefully choose a lower bound
\begin{align*}
l(\theta;x)&=\log p(x|\theta)\\
&=\log\int p(x,z|\theta)dz\\
&=\log\int q(z|x,\theta)p(x,z|\theta)/q(z|x,\theta)dz\\
&=\log E(p(x,z|\theta)/q)\\
&\ge E(log(p/q))\\
&\equiv f(q,\theta)
\end{align*}

2) E-step
Compute the posterior of latent variable, $q(z|x,\theta)=p(z|x,\theta)$.
Compute the expectation of the joint log likelihood $E(\log p(x,z|\theta))$, using $q(z|x,\theta)$ as weights.

3) M-step
Maximize the join log likelihood w.r.t. $\theta$.
'''

