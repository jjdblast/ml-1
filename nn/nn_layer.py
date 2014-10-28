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


@_register_layer_class('deno')
class DenoiseLayer(ILayer):
    '''This layer is for denoising autoencoders, so meant to be used as the first layer,
       no backprop done in this layer.'''

    def __init__(self, shape, level=0.0, **kwargs):
        self.level_ = level

    def register_updater(self, updater):
        pass

    def fprop(self):
        mask = random_generator['binomial'](1, 1-self.level_, self.in_.shape)
        self.out_ = self.in_ * mask

    def bprop(self):
        pass

    def update(self, updater):
        pass


@_register_layer_class('drop')
class DropoutLayer(ILayer):
    def __init__(self, shape, level=0.0, **kwargs):
        self.level_ = level
        self.mask_ = None

    def register_updater(self, updater):
        pass

    def fprop(self):
        tmp = random_generator['binomial'](1, 1-self.level_, (1, self.in_.shape[1]))
        self.mask_ = tmp.repeat(self.in_.shape[0], axis=0)
        self.out_ = self.in_ * self.mask_

    def bprop(self):
        self.in_ = self.out_ * self.mask_

    def update(self, updater):
        pass

@_register_layer_class('soft')
class SoftmaxLayer(ILayer):
    '''SoftmaxLayer is a pure activation layer, 
       the weights are actually in the previous connection layer.'''

    def __init__(self, shape, **kwargs):
        pass

    def register_updater(self, updater):
        pass

    def fprop(self):
        self.out_ = softmax(self.in_)

    def bprop(self):
        self.in_ = self.out_
