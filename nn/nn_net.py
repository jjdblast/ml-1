from nn_layer import *
from nn_updater import *
from nn_util import *


class Net(object):
    def __init__(self, arch, **updater_kwargs):
        self.layers_ = self.create_layers(arch)
        self.n_layers_ = len(self.layers_)
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
        for i_layer in range(self.n_layers_):
            layer = self.layers_[i_layer]
            if i_layer == 0:
                layer.in_ = x
            else:
                layer.in_ = self.layers_[i_layer-1].out_
            layer.fprop()

    def bprop(self, loss):
        for i_layer in range(self.n_layers_-1, -1, -1):
            layer = self.layers_[i_layer]
            if i_layer == self.n_layers_-1:
                layer.out_ = loss
            else:
                layer.out_ = self.layers_[i_layer+1].in_
            layer.bprop()

    def calculate_loss(self, y, loss_type=0):
        if loss_type == 0:
            # squared loss derivative.
            loss = self.layers_[-1].out_ - y
        elif loss_type == 1:
            # cross entropy derivative.
            loss = (self.layers_[-1].out_ - y) / (self.layers_[-1].out_ - self.layers_[-1].out_**2)
        return loss

    def update(self, n_iter_passed):
        self.updater_.schedule(n_iter_passed)
        for layer in self.layers_:
            layer.update(self.updater_)

    def train(self, x, y, n_epochs=10, batch_size=128, loss_type=0, x_validate=None, y_validate=None, display=False):
        if display:
            ilp = IterLinePlotter()

        n_iter_passed = 0
        for i_epoch in range(n_epochs):
            for x_batch, y_batch in data_iterator(x, y, batch_size):
                print 'iter: {}'.format(n_iter_passed)
                self.fprop(x_batch)
                loss = self.calculate_loss(y_batch, loss_type)
                self.bprop(loss)
                self.update(n_iter_passed)
                n_iter_passed += 1

                if display:
                    train_error = self.evaluate(self.predict(x_batch), y_batch)
                    ilp.update(0, ilp.count_, train_error)
                    if x_validate is not None and y_validate is not None:
                        validate_error = self.evaluate(self.predict(x_validate), y_validate)
                        ilp.update(1, ilp.count_, validate_error)
                    ilp.draw()
                    ilp.count_ += 1


    def predict(self, x):
        self.fprop(x, False)
        return self.layers_[-1].out_

    def evaluate(self, pred, y):
        n = pred.shape[0]
        return np.sum((pred - y)**2) / n * 0.5
