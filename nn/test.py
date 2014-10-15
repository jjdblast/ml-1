from scipy.io import loadmat
from ae import *
from nn_util import *
import time


def test_SAE_mnist():
    images = loadmat('../ufldl/data/IMAGES.mat')['IMAGES']
    images = np.rollaxis(images, -1, 0)
    patches = sample_patches(10000, 8, images)
    patches.shape = (-1, 8*8)

    x = normalize_3sigma(patches[:-5000])
    x_validate = normalize_3sigma(patches[-5000:])
    sae = SAE(8*8, 25, weight_decay=1e-4, sparsity_control=3, sparsity=1e-2)
    sae.train(x, x, 300, x_validate, x_validate, True)
    show_image_grid(sae.w0_.reshape(-1, 8,8))

if __name__ == '__main__':
    test_SAE_mnist()
