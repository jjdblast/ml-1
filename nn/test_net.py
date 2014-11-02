from scipy.io import loadmat
from nn_util import *
from nn_net import *
import time, pickle
from PIL import Image


def test_Net_grad_check():
    net = Net([['full', 5, {}],
               ['sig', 10, {}],
               ['full', 10, {}],
               ['sig', 5, {}]])
    x = np.random.rand(10, 5)
    y = np.random.rand(10, 5)

    for i in range(2):
        w0 = net.layers_[0].w_.copy()
        b0 = net.layers_[0].b_.copy()
        w1 = net.layers_[2].w_.copy()
        b1 = net.layers_[2].b_.copy()
        p = pack_params(w0, b0, w1, b1)
        shape = (w0.shape, b0.shape, w1.shape, b1.shape)
    
        def cost(p, x, y):
            ww0, bb0, ww1, bb1 = unpack_params(p, shape)
            z0 = np.dot(x, ww0.T) + bb0
            a0 = sigmoid(z0)
            z1 = np.dot(a0, ww1.T) + bb1
            a1 = sigmoid(z1)
            c = 0.5 / x.shape[0] * np.sum((a1-y)**2)
            return [c]
    
        grad_n = compute_numeric_grad(cost, p, x=x, y=y)
        p -= 0.1 * (grad_n + 1.0 * p)
    
        net.fprop(x)
        loss = net.calculate_loss(y)
        net.bprop(loss)
        net.update(0)
        grad = pack_params(net.layers_[0].wgrad_, net.layers_[0].bgrad_, net.layers_[2].wgrad_, net.layers_[2].bgrad_)
        p1 = pack_params(net.layers_[0].w_, net.layers_[0].b_, net.layers_[2].w_, net.layers_[2].b_)
    
        pl.plot(p)
        pl.plot(p1)
        pl.show()

def test_Net_dae():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = mnist[0][0]
    x_v = mnist[1][0]

    net = Net([['deno', 28*28, {'level': 0.3}],
               ['full', 28*28, {}],
               ['sig', 500, {}],
               ['full', 500, {}],
               ['sig', 28*28, {}]], 
              loss_type=1, base_lr=0.1, decay=0.0, momentum=0.0, update_type=0)

    print time.ctime()
    start_time = time.clock()
    net.train(x, x, n_epochs=1, batch_size=20, x_validate=x_v, y_validate=x_v, evaluate=False, display=False)
    end_time = time.clock()
    print 'time consumed: {}'.format((end_time - start_time) / 60.)
    print time.ctime()
    tmp = tile_images(net.layers_[1].w_, (28,28), (10,10))
    pl.imshow(tmp, cmap='gray')
    pl.show()

def test_Net_drop():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    images = mnist[0][0][:10000]
    x = images

    net = Net([['drop', 28*28, {'level': 0.2}],
               ['full', 28*28, {}],
               ['sig', 256, {}],
               ['drop', 256, {'level': 0.5}],
               ['full', 256, {}],
               ['sig', 28*28, {}]], 
              loss_type=1, base_lr=0.1, decay=0.0, momentum=0.0, update_type=0)

    net.train(x, x, n_epochs=1, batch_size=1, evaluate=False, display=False)
    tmp = tile_images(net.layers_[1].w_, (28,28), (10,10))
    save_image(tmp, 'pic/nn_dropout_ent_lr0.1_epoch1_batch1.png')

def test_Net_dropsoft():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    images = mnist[0][0][:15000]
    labels = mnist[0][1][:15000]
    x = images[:10000]
    y = labels[:10000]
    x_v = images[10000:]
    y_v = labels[10000:]

    net = Net([['drop', 28*28, {'level': 0.2}],
               ['full', 28*28, {}],
               ['relu', 500, {}],
               ['drop', 500, {'level': 0.5}],
               ['full', 500, {}],
               ['soft', 10, {}]],
              loss_type=2, base_lr=0.1, decay=0.0, momentum=0.0, update_type=0)
    net.train(x, y, n_epochs=30, batch_size=20, display=False, evaluate=True, x_validate=x_v, y_validate=y_v, shuffle=True)

def test_Net_soft():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = mnist[0][0]
    y = mnist[0][1]
    x_v = mnist[1][0]
    y_v = mnist[1][1]

    net = Net([['full', 28*28, {}],
               ['sig', 500, {}],
               ['full', 500, {}],
               ['soft', 10, {}]],
              loss_type=2, base_lr=0.1, decay=0, update_type=0)
    start_time = time.clock()
    net.train(x, y, n_epochs=3, batch_size=20, evaluate=False, shuffle=False,
              x_validate=x_v, y_validate=y_v)
    end_time = time.clock()
    print (end_time - start_time) / 60.

def test_soft():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    x = mnist[0][0][:10000]
    y = mnist[0][1][:10000]
    x_v = mnist[1][0][:10000]
    y_v = mnist[1][1][:10000]

    net = Net([['full', 28*28, {}],
               ['soft', 10, {}]],
              loss_type=2, base_lr=0.1, decay=0.0, momentum=0.0, update_type=0)
    net.train(x, y, n_epochs=2, batch_size=30, evaluate=True, shuffle=True,
              x_validate=x_v, y_validate=y_v)


if __name__ == '__main__':
    test_Net_dae()
