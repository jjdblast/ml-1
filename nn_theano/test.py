import operator, time, pickle

from layer import *
from net import *

import theano
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'


def test_full():
    fc0 = FullConnectLayer((5, 10))
    fc1 = FullConnectLayer((15, 5))
    fc2 = FullConnectLayer((5, 15))
    
    # fprop
    x = tensor.dmatrix('x')
    xz0 = fc0.fprop(x)
    xz1 = fc1.fprop(xz0)
    xz2 = fc2.fprop(xz1)

    # bprop
    y = tensor.dmatrix('y')
    yz2 = fc2.bprop(xz2 - y)
    yz1 = fc1.bprop(yz2)
    yz0 = fc0.bprop(yz1)

    updates = reduce(operator.add, [fc0.update(), fc1.update(), fc2.update()])

    # call `function` to compile a graph, throw everything into this graph (inputs, outputs, updates).
    # if compile graphs seperately for `fprop`, `bprop`, then some of the inputs will be missing.
    output = function([x, y], [xz2, yz0], updates=updates)
    output(np.random.rand(2, 10), np.random.rand(2, 5))
    print fc0.b_.get_value(), fc1.b_.get_value(), fc2.b_.get_value()

def test_full_soft():
    fc0 = FullConnectLayer((5, 2))
    soft1 = SoftmaxLayer((5, 5))

    x, y = tensor.dmatrices('x', 'y')

    # fprop
    xz0 = fc0.fprop(x)
    xz1 = soft1.fprop(xz0)

    # bprop
    yz1 = soft1.bprop(xz1 - y)
    yz0 = fc0.bprop(yz1)

    updates = reduce(operator.add, [fc0.update(), soft1.update()])
    output = function([x, y], [xz1, yz0], updates=updates, on_unused_input='ignore')
    output(np.random.rand(2,2), np.random.rand(2, 5))
    print fc0.w_.get_value(), fc0.b_.get_value()

def test_net_soft():
    net = Net([['full', 2, {}], ['soft', 2, {}]])

    x, y = tensor.dmatrices('x', 'y')
    xzn = net.fprop(x)
    yz0 = net.bprop(y)
    updates = net.update()
    output = function([x,y], [xzn, yz0], updates=updates)
    output(np.random.rand(2,2), np.random.rand(2,2))
    print net.layers_[0].w_.get_value(), net.layers_[0].b_.get_value()

def test_net_sigm():
    net = Net([['full', 3, {}], ['sigm', 2, {}]])

    x, y = tensor.dmatrices('x', 'y')
    xz1 = net.fprop(x)
    yz0 = net.bprop(y)

    output = function([x, y], [xz1, yz0])
    output(np.random.rand(1,3), np.random.rand(1, 2))
    print net.layers_[0].w_.get_value()

def test_full_sigm():
    fc0 = FullConnectLayer((2, 3))
    sig1 = SigmoidLayer((2, 2))

    x, y = tensor.dmatrices('x', 'y')

    # fprop
    xz0 = fc0.fprop(x)
    xz1 = sig1.fprop(xz0)

    # bprop
    yz1 = sig1.bprop(y)
    yz0 = fc0.bprop(yz1)

    output = function([x, y], [xz1, yz0])
    output(np.random.rand(1,3), np.random.rand(1, 2))
    print fc0.w_.get_value(), fc0.b_.get_value()

def test_net_dae():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = shared(mnist[0][0].astype(np.float64), borrow=True)
    x_v = shared(mnist[1][0].astype(np.float64), borrow=True)

    net = Net([['deno', 28*28, {'level': 0.3}],
               ['full', 28*28, {}],
               ['sigm', 500, {}],
               ['full', 500, {}],
               ['sigm', 28*28, {}]],
              loss_type='ceml')

    print time.ctime()
    net.train(x, x, x_v=x_v, y_v=x_v, n_epochs=15, batch_size=20)
    print time.ctime()
    filters = tile_images(net.layers_[1].w_.get_value(),
                          (28, 28), (10, 10))
    save_image(filters, 'pic/dae.png')

def test_net_soft():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = shared(mnist[0][0].astype(np.float64), borrow=True)
    y = shared(mnist[0][1].astype(np.int32), borrow=True)
    x_v = shared(mnist[1][0].astype(np.float64), borrow=True)
    y_v = shared(mnist[1][1].astype(np.int32), borrow=True)
    x_t = shared(mnist[2][0].astype(np.float64), borrow=True)
    y_t = shared(mnist[2][1].astype(np.int32), borrow=True)

    net = Net([['full', 28*28, {}],
               ['sigm', 100, {}],
               ['full', 100, {}],
               ['soft', 10, {}]],
              loss_type='cemc',
              updater_args={'base_lr': 0.1})
    net.train(x, y, x_v=x_v, y_v=y_v, n_epochs=15, batch_size=20)
    pred = net.predict(x_t)
    rate = tensor.mean(tensor.neq(pred, y_t))
    rate_f = function([], rate, givens={x_t: x_t, y_t: y_t})
    print rate_f()

def test_net_dropout_ae():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = shared(mnist[0][0].astype(np.float64), borrow=True)
    x_v = shared(mnist[1][0].astype(np.float64), borrow=True)

    net = Net([['drop', 28*28, {'level': 0.2}],
               ['full', 28*28, {}],
               ['sigm', 128, {}],
               ['drop', 128, {'level': 0.5}],
               ['full', 128, {}],
               ['sigm', 28*28, {}]],
              loss_type='ceml',
              updater_args={'base_lr': 0.1})
    print time.ctime()
    net.train(x, x, x_v=x_v, y_v=x_v, n_epochs=10, batch_size=20)
    print time.ctime()
    filters = tile_images(net.layers_[1].w_.get_value(),
                          (28, 28), (10, 10))
    save_image(filters, 'pic/dae.png')


if __name__ == '__main__':
    test_net_dropout_ae()
