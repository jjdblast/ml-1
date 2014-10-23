from scipy.io import loadmat
from ae import *
from nn_util import *
import time, pickle
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

if __name__ == '__main__':
    test_DAE_mnist()
