from theano import tensor
from util import *


LOSS = {}
ERROR = {}

@register(LOSS, 'mse')
def mse_loss(pred, target):
    return tensor.mean((pred - target)**2, axis=0).sum()

register(ERROR, 'mse')(mse_loss)

@register(LOSS, 'ceml')
def ceml_loss(pred, target):
    L = - tensor.sum(target * tensor.log(pred) + (1-target) * tensor.log(1-pred), axis=1)
    L = tensor.mean(L)
    return L

register(ERROR, 'ceml')(ceml_loss)

@register(LOSS, 'cemc')
def cemc_loss(pred, target):
    L = - tensor.mean(tensor.log(pred[tensor.arange(target.size), target]))
    return L

@register(ERROR, 'cemc')
def cemc_error(pred, target):
    error = tensor.mean(tensor.neq(pred.argmax(axis=1), target))
    return error
