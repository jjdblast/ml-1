from layer import *
from updater import *
from metric import *

import time


class Net(object):
    def __init__(self, arch, loss_type='mse', updater_args={'base_lr': 0.1, 'momentum': 0.0, 'l2_decay': 0.0}):
        self.loss_type_ = loss_type
        self.layers_ = self.create_layer(arch)
        self.updater_ = SGDUpdater(**updater_args)
        self.register_updater()

    def create_layer(self, arch):
        n = len(arch)
        layers = []
        self.params_ = []
        for i in range(n):
            shape = (arch[i+1][1] if i < n-1 else arch[i][1],
                     arch[i][1])
            layer = LAYER[arch[i][0]](shape, **arch[i][2])
            layers.append(layer)
            if layer.params_ is not None:
                self.params_.extend(layer.params_)
        return layers

    def register_updater(self):
        self.params_updater_id_ = []
        for param in self.params_:
            self.params_updater_id_.append(self.updater_.register_updater(param))

    def fprop(self, in_, is_train=True):
        for layer in self.layers_:
            in_ = layer.fprop(in_, is_train)
        return in_

    def bprop(self, loss):
        self.params_g_ = [tensor.grad(loss, param) for param in self.params_]
        return self.params_g_

    def update(self, i_epoch):
        self.updater_.tune_lr(i_epoch)
        updates = self.updater_.update(zip(self.params_, self.params_g_, self.params_updater_id_))
        return updates

    def train(self, x_raw, y_raw, x_v_raw=None, y_v_raw=None, n_epochs=1, batch_size=1):
        index = tensor.iscalar('index')
        in_, out_ = tensor.dmatrices('in_', 'out_')
        if self.loss_type_ == 'cemc':
            out_ = tensor.ivector('out_')

        pred = self.fprop(in_)
        loss = LOSS[self.loss_type_](pred, out_)
        self.bprop(loss)

        # for training
        epoch = tensor.iscalar('epoch')
        updates = self.update(epoch)
        x = shared(x_raw, borrow=True)
        y = shared(y_raw, borrow=True)

        train_f = function([index, epoch], loss, updates=updates, on_unused_input='ignore',
                           givens={in_: x[index : index + batch_size],
                                   out_: y[index : index + batch_size]}
                  )
        n = x.get_value().shape[0]

        # for validation
        is_validate = (x_v_raw is not None and y_v_raw is not None)
        if is_validate:
            v_pred = self.fprop(in_, is_train=False)
            v_error = ERROR[self.loss_type_](v_pred, out_)
            x_v = shared(x_v_raw, borrow=True)
            y_v = shared(y_v_raw, borrow=True)

            validate_f = function([index], v_error, on_unused_input='ignore',
                                  givens={in_: x_v[index : index + batch_size],
                                          out_: y_v[index : index + batch_size]}
                         )
            n_v = x_v.get_value().shape[0]

        start_time = time.clock()
        for i_epoch in xrange(n_epochs):
            for i_batch in xrange(0, n, batch_size):
                train_f(i_batch, i_epoch)

            if is_validate:
                errors = [validate_f(i) for i in xrange(0, n_v, batch_size)]
                print 'epoch: {0}, validation error: {1}'.format(i_epoch, np.mean(errors))

        end_time = time.clock()
        print 'time consumed: {}'.format((end_time - start_time) / 60.)

    def predict(self, x_raw):
        x = tensor.dmatrix('x')
        pred = self.fprop(x, is_train=False)
        if self.loss_type_ == 'cemc':
            pred = pred.argmax(axis=1)
        pred_f = function([x], pred)
        return pred_f(x_raw)
