import numpy.distutils.__config__  # necessary for theano in Win7

import abc
from theano import tensor, function, shared
from util import *


LAYER = {}

class ILayer(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, *args, **kwargs):
        pass

    def fprop(self, *args, **kwargs):
        pass


@register(LAYER, 'full')
class FullConnectLayer(ILayer):
    def __init__(self, shape):
        w = init_weights(shape)
        self.w_ = shared(value=w, name='w', borrow=True)
        b = np.zeros(shape[0])
        self.b_ = shared(value=b, name='b', borrow=True)
        self.params_ = [self.w_, self.b_]

    def fprop(self, in_, is_train=True):
        self.out_ = tensor.dot(in_, self.w_.T) + self.b_
        return self.out_


@register(LAYER, 'sigm')
class SigmoidLayer(ILayer):
    def __init__(self, shape):
        self.params_ = None

    def fprop(self, in_, is_train=True):
        self.out_ = tensor.nnet.sigmoid(in_)
        return self.out_


@register(LAYER, 'soft')
class SoftmaxLayer(ILayer):
    def __init__(self, shape):
        self.params_ = None

    def fprop(self, in_, is_train=True):
        self.out_ = tensor.nnet.softmax(in_)
        return self.out_


@register(LAYER, 'deno')
class DenoiseLayer(ILayer):
    def __init__(self, shape, level=0.0, noise='binomial', seed=123):
        self.level_ = level
        self.noise_ = noise
        self.rng_ = tensor.shared_randomstreams.RandomStreams(seed)
        self.params_ = None

    def fprop(self, in_, is_train=True):
        if self.noise_ == 'binomial':
            self.mask_ = self.rng_.binomial(size=in_.shape, n=1, p=1-self.level_)
            self.out_ = in_ * self.mask_
        elif self.noise_ == 'gaussian':
            self.mask_ = self.rng_.normal(size=in_.shape, std=self.level_)
            self.out_ = in_ + self.mask_
        return self.out_


@register(LAYER, 'drop')
class DropoutLayer(ILayer):
    def __init__(self, shape, level=0.0, seed=123):
        self.level_ = level
        self.rng_ = tensor.shared_randomstreams.RandomStreams(seed)
        self.params_ = None

    def fprop(self, in_, is_train=True):
        if is_train:
            self.mask_ = self.rng_.binomial(size=in_.shape, n=1, p=1-self.level_)
            self.out_ = in_ * self.mask_
        else:
            self.out_ = in_ * (1-self.level_)
        return self.out_


@register(LAYER, 'relu')
class ReLULayer(ILayer):
    def __init__(self, shape):
        self.params_ = None

    def fprop(self, in_, is_train=True):
        self.out_ = tensor.maximum(in_, 0.0)
        return self.out_
