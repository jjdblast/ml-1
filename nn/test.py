from scipy.io import loadmat
from ae import *
from nn_util import *


def test_SAE_mnist():
    images = loadmat('../ufldl/data/IMAGES.mat')['IMAGES']
    images = np.rollaxis(images, -1, 0)
    patches = sample_patches(10000, 8, images)
    patches.shape = (-1, 8*8)

    x = normalize_3sigma(patches)
    sae = SAE(8*8, 25, weight_decay=1e-4, sparsity_control=3, sparsity=1e-2)
    res = sae.train(x, x)
    w0 = res.x[:8*8*25].reshape(25, 8, 8)
    show_image_grid(w0)


if __name__ == '__main__':
    test_SAE_mnist()
