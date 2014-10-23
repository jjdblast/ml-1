from scipy.io import loadmat
from ae import *
from nn_util import *
from nn_net import *
import time, pickle
from utils import *
from PIL import Image


def test_SAE_mnist():
    images = loadmat('../ufldl/data/IMAGES.mat')['IMAGES']
    images = np.rollaxis(images, -1, 0)
    patches = sample_patches(10000, 8, images)
    patches.shape = (-1, 8*8)

    x = normalize_3sigma(patches)
    sae = SAE(8*8, 25, weight_decay=1e-4, sparsity_control=3, sparsity=1e-2)
    sae.train(x, x, 400, x_validate=None, y_validate=None, display=False)
    tmp = tile_images(sae.w0_, (8,8), (5,5), (1,1))
    pl.imshow(tmp, cmap='gray')
    pl.show()

def test_DAE_mnist():    
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    images = mnist[0][0]
    x = images

    dae = DAE(28*28, 500, False)
    dae.train(x, x.copy(), level=0.3, n_epochs=1, batch_size=20)
    tmp = tile_images(dae.w0_, (28,28), (10,10), (1,1))
    pl.imshow(tmp, cmap='gray')
    pl.show()

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

def test_Net_ae():
    with open('../ufldl/data/mnist.pkl', 'rb') as fp:
        mnist = pickle.load(fp)
    images = mnist[0][0]
    x = images

    net = Net([['deno', 28*28, {'level': 0.3}],
               ['full', 28*28, {}],
               ['sig', 500, {}],
               ['full', 500, {}],
               ['sig', 28*28, {}]], base_lr=0.1, decay=0.0, momentum=0.0)

    net.train(x, x, n_epochs=15, batch_size=20, loss_type=1, update_type=0, 
                x_validate=None, y_validate=None, evaluate=False, display=False)
    tmp = tile_images(net.layers_[1].w_, (28,28), (10,10))
    pl.imshow(tmp, cmap='gray')
    pl.show()


if __name__ == '__main__':
    test_SAE_mnist()
