from nn_util import *


class SAE(object):
    '''Sparse autoencoder, using lbfgs to train.'''

    def __init__(self, input_size, hidden_size, weight_decay=1.0, sparsity_control=0.0, sparsity=0.1):
        self.w0_ = init_weights((hidden_size, input_size))
        self.w1_ = init_weights((input_size, hidden_size))
        self.b0_ = np.zeros(hidden_size)
        self.b1_ = np.zeros(input_size)
        self.weight_decay_ = weight_decay
        self.sparsity_control_ = sparsity_control
        self.sparsity_ = sparsity

    def get_cost_grad(self, x, y):
        # fprop
        a0 = x
        z0 = np.dot(a0, self.w0_.T) + self.b0_
        a1 = sigmoid(z0)
        z1 = np.dot(a1, self.w1_.T) + self.b1_
        a2 = sigmoid(z1)

        n_examples = x.shape[0]
        sparsity_hat = a1.mean(axis=0)
        cost = (0.5 / n_examples) * np.sum((a2 - y)**2) + \
               (0.5 * self.weight_decay_) * \
               np.sum(np.sum(self.w0_**2) + np.sum(self.w1_**2)) + \
               self.sparsity_control_ * np.sum(kl(self.sparsity_, sparsity_hat))

        # bprop
        '''
        \nabla_{a2}cost = 1.0/n_examples * \Sigma(a2-y),
        \nabla_{w1}a2 = a2*(1-a2) * a1,
        \nabla_{a1}a2 = a2*(1-a2) * w1
        \nabla_{w0}a1 = a1*(1-a1) * a0,

        \nabla_{w0}kl = \nabla_{a1}kl * \nabla_{w0}a1

        \nabla{w1}cost = \nabla_{a2}cost * \nabla_{w1}a2  + weight_decay*w1,
        \nabla{w0}cost = \nabla_{a2}cost * \nabla_{a1}a2 * \nabla_{w0}{a1} + weight_decay*w0 + sparsity_control*\nabla_{w0}kl,
        '''
        d2 = (a2 - y) * a2 * (1 - a2)
        w1_grad = np.dot(d2.T, a1) / n_examples + \
                    self.weight_decay_ * self.w1_
        b1_grad = d2.mean(axis=0)
        d1 = (np.dot(d2, self.w1_) + self.sparsity_control_ * \
                kl_prime(self.sparsity_, sparsity_hat)) * a1 * (1 - a1)
        w0_grad = np.dot(d1.T, a0) / n_examples + \
                    self.weight_decay_ * self.w0_
        b0_grad = d1.mean(axis=0)
        return cost, (w0_grad, w1_grad, b0_grad, b1_grad)

    def _cost_wraper(self, params, x=None, y=None):
        shapes = [self.w0_.shape, self.w1_.shape, self.b0_.shape, self.b1_.shape]
        self.w0_, self.w1_, self.b0_, self.b1_ = unpack_params(params, shapes)
        cost, unpacked_grads = self.get_cost_grad(x, y)
        grads = pack_params(*unpacked_grads)
        return cost, grads

    def train(self, x, y, n_epochs=400, x_validate=None, y_validate=None, display=False):
        _plot = None
        if display:
            ilp = IterLinePlotter()
            _plot = lambda p: self._display(p, x, y, x_validate, y_validate, ilp)

        params = pack_params(self.w0_, self.w1_, self.b0_, self.b1_)
        res = optimize.minimize(self._cost_wraper, params, args=(x, y),
                    method='L-BFGS-B', jac=True, 
                    options={'maxiter': n_epochs, 'iprint': True},
                    callback=_plot)

        shapes = [self.w0_.shape, self.w1_.shape, self.b0_.shape, self.b1_.shape]
        self.w0_, self.w1_, self.b0_, self.b1_ = unpack_params(res.x, shapes)

    def _reconstruction_error(self, x, y):
        a0 = x
        z0 = np.dot(a0, self.w0_.T) + self.b0_
        a1 = sigmoid(z0)
        z1 = np.dot(a1, self.w1_.T) + self.b1_
        a2 = sigmoid(z1)
        n_examples = x.shape[0]
        error = 0.5 / n_examples * np.sum((a2 - y)**2)
        return error

    def _display(self, params, x, y, x_validate, y_validate, ilp):
        cost = self._reconstruction_error(x, y)
        ilp.update(1, ilp.count_, cost)
        if x_validate is not None and y_validate is not None:
            cost_validate = self._reconstruction_error(x_validate, y_validate)
            ilp.update(2, ilp.count_, cost_validate)
        ilp.draw()
        ilp.count_ += 1


class DAE(object):
    '''Denoising autoencoder, using sgd to train.'''

    def __init__(self, input_size, hidden_size, tied=False):
        '''If `tied` is True, then used tied weights.'''

        self.w0_ = init_weights((hidden_size, input_size))
        self.b0_ = np.zeros(hidden_size)

        self.tied_ = tied
        self.w1_ = self.w0_.T if self.tied_ else init_weights((input_size, hidden_size))
        self.b1_ = np.zeros(input_size)
        self.rng_ = np.random.RandomState(123)

    def corrupt(self, x, level):
        return self.rng_.binomial(n=1, p=1.0-level, size=x.shape) * x

    def get_cost_grad(self, x, y, level):
        a0 = self.corrupt(x, level)
        z0 = np.dot(a0, self.w0_.T) + self.b0_
        a1 = sigmoid(z0)
        z1 = np.dot(a1, self.w1_.T) + self.b1_
        a2 = sigmoid(z1)

        n_examples = x.shape[0]
        cost = - (y * np.log(a2) + (1-y) * np.log(1-a2)).sum() / n_examples
        
        '''If the weights are tied, then could we still back propagate the error
        the usual way? In untied cases, we could do it that way because when back
        propagating error from ith layer to (i-1)th layer, we just hold weights in
        ith layer constant, and that's ok if the weights are not tied. However, 
        if weights are tied, we could not do that.
        Fortunately, we could do it that way, it can be proved. When updating untied
        weights, we should sum up the two gradient matrices.'''

        d2 = (a2 - y)
        w1_grad = np.dot(d2.T, a1) / n_examples
        b1_grad = d2.mean(axis=0)
        d1 = np.dot(d2, self.w1_) * a1 * (1 - a1)
        w0_grad = np.dot(d1.T, a0) / n_examples
        b0_grad = d1.mean(axis=0)
        return w0_grad, w1_grad, b0_grad, b1_grad

    def train(self, x, y, level=0.0, lr=0.1, n_epochs=10, batch_size=30):
        for i_epoch in range(n_epochs):
            print 'epoch: {}'.format(i_epoch)
            for x_batch, y_batch in data_iterator(x, y, batch_size):
                w0_grad, w1_grad, b0_grad, b1_grad = self.get_cost_grad(x_batch, y_batch, level)
                if self.tied_:
                    w_grad = w0_grad + w1_grad.T
                    self.w0_ -= lr * w_grad
                    self.w1_ = self.w0_.T
                else:
                    self.w0_ -= lr * w0_grad
                    self.w1_ -= lr * w1_grad
                self.b0_ -= lr * b0_grad
                self.b1_ -= lr * b1_grad
