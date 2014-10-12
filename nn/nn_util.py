import numpy as np
from scipy import optimize
import pylab as pl


random_generator = {
    'uniform': np.random.uniform,
}

def init_weights(shape, eps=None, random_type='uniform'):
    m, n = shape
    if eps is None:
        eps = np.sqrt(6) / np.sqrt(m + n + 1)
    return random_generator[random_type](low=-eps, high=eps, size=shape)

def pack_params(*params):
    return np.concatenate([param.ravel() for param in params])

def unpack_params(params, shapes=((10,5), (5,10), (10,), (5,))):
    unpacked_params = []
    borders = np.cumsum([reduce(np.multiply, shape) for shape in shapes])
    i = 0
    for j in range(len(borders)):
        param = params[i:borders[j]].reshape(shapes[j])
        unpacked_params.append(param)
        i = borders[j]
    return unpacked_params

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def kl(x, x_hat):
    return x * np.log(x / x_hat) + (1-x) * np.log((1-x) / (1-x_hat))

def kl_prime(x, x_hat):
    '''
    \nabla{x_hat}kl = -x/x_hat + (1-x)/(1-x_hat),
    '''
    return -x / x_hat + (1-x) / (1-x_hat)

def data_iterator(x, y, n_epochs=2, batch_size=256, shuffle_on=True):
    if shuffle_on:
        rng = np.random.RandomState()

    for i in range(n_epochs):
        if shuffle_on:
            seed = np.random.randint(1000)
            rng.seed(seed)
            rng.shuffle(x)
            rng.seed(seed)
            rng.shuffle(y)
        for j in range(0, x.shape[0], batch_size):
            yield x[j:j+batch_size], y[j:j+batch_size], i

def sgd(cost_func, initial_guess, x, y, n_epochs=100, batch_size=256, lr=0.1, lr_decay=None, tol=1e-5, verbose=True, display=True):
    params = initial_guess
    cost_ = 0
    iter_counter = 0
    for x_, y_, i_epoch in data_iterator(x, y, n_epochs, batch_size):
        cost, grads = cost_func(params, x_, y_)
        if np.abs(cost - cost_) < tol:
            break
        params -= lr * grads
        cost_ = cost
        if verbose:
            print 'iter: {0},\tcost: {1}'.format(iter_counter, cost_)
        if display:

        iter_counter += 1
    return {'x': params}


def compute_numeric_grad(cost_func, params, eps=1e-5, **kwargs):
    numeric_grad = np.empty(shape=params.shape)
    for i in xrange(len(params)):
        params[i] -= eps
        ret0 = cost_func(params, **kwargs)
        params[i] += 2 * eps
        ret1 = cost_func(params, **kwargs)
        numeric_grad[i] = (ret1[0] - ret0[0]) / (2 * eps)
    return numeric_grad

def sample_patches(n_patches, patch_size, images):
    n_images, r_image, c_image = images.shape
    half = patch_size / 2
    sampled_n = np.random.randint(n_images, size=n_patches)
    sampled_r = np.random.randint(low=half, high=r_image-half, size=n_patches)
    sampled_c = np.random.randint(low=half, high=c_image-half, size=n_patches)
    grid = np.ogrid[-half:half, -half:half]
    patches = np.array([images[sampled_n[i], sampled_r[i]+grid[0], sampled_c[i]+grid[1]] for i in range(n_patches)])
    return patches

def normalize_3sigma(x):
    x -= x.mean(axis=0)
    std_3 = 3.0 * np.std(x.ravel(), ddof=1)
    x = np.maximum(np.minimum(x, std_3), -std_3) / std_3
    x = (x + 1) * 0.4 + 0.1
    return x

def show_image_grid(x):
    n, w, h = x.shape
    r = int(np.ceil(np.sqrt(n)))
    c = int(np.ceil(n / float(r)))
    fig, ax = pl.subplots(r, c)
    for i in range(r):
        for j in range(c):
            k = i*c + j
            if k >= n:
                break
            ax[i,j].imshow(x[k], cmap='gray')
    pl.show()

def 
