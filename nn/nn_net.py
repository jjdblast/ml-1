from nn_layer import *
from nn_updater import *
from nn_util import *


class Net(object):
    def __init__(self, arch, loss_type=0, **updater_kwargs):
        self.layers_ = self.create_layers(arch)
        self.n_layers_ = len(self.layers_)
        self.loss_type_ = loss_type
        self.updater_ = SGDUpdater(**updater_kwargs)
        self.register_updater()

    def create_layers(self, arch):
        n = len(arch)
        layers = []
        for i in range(n):
            shape = (arch[i+1][1] if i < n-1 else arch[i][1], arch[i][1])
            layer = layer_class[arch[i][0]](shape, **arch[i][2])
            layers.append(layer)
        return layers

    def register_updater(self):
        for layer in self.layers_:
            layer.register_updater(self.updater_)

    def fprop(self, x, is_train=True):
        '''`is_train` for dropout layers.'''

        for i_layer in range(self.n_layers_):
            layer = self.layers_[i_layer]
            if i_layer == 0:
                layer.in_ = x
            else:
                layer.in_ = self.layers_[i_layer-1].out_
            layer.fprop(is_train)

    def bprop(self, loss):
        for i_layer in range(self.n_layers_-1, -1, -1):
            layer = self.layers_[i_layer]
            if i_layer == self.n_layers_-1:
                layer.out_ = loss
            else:
                layer.out_ = self.layers_[i_layer+1].in_
            layer.bprop()

    def calculate_loss(self, y):
        '''For `loss_type=2`, make sure the labels are arranged within [0,K].'''
        # mse.
        if self.loss_type_ == 0:
            loss = self.layers_[-1].out_ - y
        # cross entropy, multilabel Bernoulli.
        elif self.loss_type_ == 1:
            loss = (self.layers_[-1].out_ - y) / (self.layers_[-1].out_ - self.layers_[-1].out_**2 + 1e-8)
        # cross entropy, multiclass.
        elif self.loss_type_ == 2:
            # TODO: check dim of y.
            tmp = np.zeros(self.layers_[-1].out_.shape)
            tmp[np.arange(y.size), y] = 1
            loss = self.layers_[-1].out_ - tmp
        return loss

    def update(self, n_iter_passed):
        self.updater_.schedule(n_iter_passed)
        for layer in self.layers_:
            layer.update(self.updater_)

    def train(self, x, y, n_epochs=10, batch_size=128, x_validate=None, 
                y_validate=None, evaluate=False, display=False, shuffle=False):
        '''With `display` being True, train error and validation error 
           will be plotted via `ilp`, and it will make the training process
           slower, you can close the figure at any time to speed up the 
           training process.'''

        if evaluate:
            n_eval = 0
            train_error_sum = 0
            if x_validate is not None and y_validate is not None:
                validate_error_sum = 0

        if display:
            ilp = IterLinePlotter()

        n_iter_passed = 0
        for i_epoch in range(n_epochs):
            print 'epoch: {}'.format(i_epoch)
            for x_batch, y_batch in data_iterator(x, y, batch_size, shuffle):
                self.fprop(x_batch)
                loss = self.calculate_loss(y_batch)
                self.bprop(loss)
                self.update(n_iter_passed)

                if evaluate:
                    n_eval += 1
                    train_error = self.evaluate(x_batch, y_batch)
                    train_error_sum += train_error
                    if x_validate is not None and y_validate is not None:
                        validate_error = self.evaluate(x_validate, y_validate)
                        validate_error_sum += validate_error

                if display:
                    train_error = self.evaluate(x_batch, y_batch)
                    ilp.update(0, ilp.count_, train_error)
                    if x_validate is not None and y_validate is not None:
                        validate_error = self.evaluate(x_validate, y_validate)
                        ilp.update(1, ilp.count_, validate_error)
                    ilp.draw()
                    ilp.count_ += 1

                n_iter_passed += 1

            if evaluate:
                msg = 'train error: {}'.format(train_error_sum / n_eval)
                if x_validate is not None and y_validate is not None:
                    msg += ', validate error: {}'.format(validate_error_sum / n_eval)
                print msg

    def predict(self, x):
        self.fprop(x, False)
        pred = self.layers_[-1].out_
        if self.loss_type_ == 0:
            return pred
        elif self.loss_type_ == 1:
            return pred
        elif self.loss_type_ == 2:
            return pred.argmax(axis=1)

    def evaluate(self, x, y):
        '''Calculate different metrics for different loss_type. For now,
           `loss_type=0`: mse, calculate reconstruction error.
           `loss_type=1`: multilabel Bernoulli cross entropy, calculate reconstruction error.
           `loss_type=2`: multiclass cross entropy, calculate accuracy.
        '''
        self.fprop(x, False)
        pred = self.layers_[-1].out_
        n = pred.shape[0]
        result = 0
        if self.loss_type_ == 0:
            result = np.sum((pred - y)**2) / n
        elif self.loss_type_ == 1:
            result = np.sum((pred - y)**2) / n
        elif self.loss_type_ == 2:
            idx = pred.argmax(axis=1)
            result = (idx == y).sum() / float(y.size)
        return result
