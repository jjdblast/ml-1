import operator, time, pickle

from layer import *
from net import *

import theano
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'


def test_net_dae():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = mnist[0][0].astype(np.float64)
    x_v = mnist[1][0].astype(np.float64)

    net = Net([['deno', 28*28, {'level': 0.3}],
               ['full', 28*28, {}],
               ['sigm', 500, {}],
               ['full', 500, {}],
               ['sigm', 28*28, {}]],
              loss_type='ceml')

    print time.ctime()
    net.train(x, x, x_v_raw=x_v, y_v_raw=x_v, n_epochs=10, batch_size=20)
    print time.ctime()
    filters = tile_images(net.layers_[1].w_.get_value(),
                          (28, 28), (10, 10))
    save_image(filters, 'pic/dae.png')

def test_net_soft():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = mnist[0][0].astype(np.float64)
    y = mnist[0][1].astype(np.int32)
    x_v = mnist[1][0].astype(np.float64)
    y_v = mnist[1][1].astype(np.int32)

    net = Net([['drop', 28*28, {'level': 0.2}],
               ['full', 28*28, {}],
               ['sigm', 100, {}],
               ['full', 100, {}],
               ['soft', 10, {}]],
              loss_type='cemc',
              updater_args={'base_lr': 0.1})
    net.train(x, y, x_v_raw=x_v, y_v_raw=y_v, n_epochs=15, batch_size=20)

    x_t = mnist[2][0].astype(np.float64)
    y_t = mnist[2][1].astype(np.int32)
    pred = net.predict(x_t)
    rate = (pred != y_t).sum() / float(y_t.size)

def test_net_dropout_ae():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = mnist[0][0].astype(np.float64)
    x_v = mnist[1][0].astype(np.float64)

    net = Net([['drop', 28*28, {'level': 0.2}],
               ['full', 28*28, {}],
               ['sigm', 128, {}],
               ['drop', 128, {'level': 0.5}],
               ['full', 128, {}],
               ['sigm', 28*28, {}]],
              loss_type='ceml',
              updater_args={'base_lr': 0.1})
    print time.ctime()
    net.train(x, x, x_v_raw=x_v, y_v_raw=x_v, n_epochs=10, batch_size=20)
    print time.ctime()
    filters = tile_images(net.layers_[1].w_.get_value(),
                          (28, 28), (10, 10))
    save_image(filters, 'pic/dae.png')


if __name__ == '__main__':
    test_net_dropout_ae()
