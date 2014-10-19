import abc
from nn_util import *


layer_class = {}

def _register_layer_class(name):
    def decorator(cls):
        layer_class[name] = cls
        return cls
    return decorator

class ILayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def register_updater(self, updater):
        return

    @abc.abstractmethod
    def fprop(self):
        return

    @abc.abstractmethod
    def bprop(self):
        return

    @abc.abstractmethod
    def update(self, updater):
        return


@_register_layer_class('full')
class FullConnectLayer(ILayer):
    def __init__(self, shape, **kwargs):
        self.w_ = init_weights(shape)
        self.b_ = np.zeros(shape[0])
        self.wgrad_ = None
        self.bgrad_ = None
        self.in_ = None
        self.out_ = None

    def fprop(self):
        self.out_ = np.dot(self.in_, self.w_.T) + self.b_

    def bprop(self):
        # partial derivative over weights.
        self.wgrad_ = np.dot(self.out_.T, self.in_) / self.out_.shape[0]
        self.bgrad_ = self.out_.mean(axis=0)
        # partial derivative over activations.
        self.in_ = np.dot(self.out_, self.w_)

    def register_updater(self, updater):
        self.w_updater_id_ = updater.register(self.w_)
        self.b_updater_id_ = updater.register(self.b_)

    def update(self, updater):
        self.w_ = updater.update(self.w_updater_id_, self.w_, self.wgrad_)
        self.b_ = updater.update(self.b_updater_id_, self.b_, self.bgrad_)


@_register_layer_class('sig')
class SigmoidLayer(ILayer):
    def __init__(self, shape, **kwargs):
        pass

    def register_updater(self, updater):
        pass

    def fprop(self):
        self.out_ = sigmoid(self.in_)
        # no need to keep input in bprop for sigmoid layer.
        self.in_ = self.out_

    def bprop(self):
        self.in_ = self.out_ * self.in_ * (1 - self.in_)

    def update(self, updater):
        pass


@_register_layer_class('relu')
class ReLULayer(ILayer):
    def __init__(self, shape, **kwargs):
        pass

    def register_updater(self, updater):
        pass

    def fprop(self):
        self.out_ = np.maximum(0, self.in_)
    
    def bprop(self):
        d = np.zeros(self.in_.shape)
        d[self.in_>0] = 1
        self.in_ = self.out_ * d

    def update(self, updater):
        pass
