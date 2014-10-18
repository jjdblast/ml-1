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

def sgd(cost_func, initial_guess, x, y, n_epochs=100, batch_size=1024, lr=0.025, lr_decay=None, tol=1e-5, verbose=False, display=False, callback=None):
    params = initial_guess
    cost_ = 0
    iter_counter = 0
    if display:
        ilp = IterLinePlotter()

    for x_, y_, i_epoch in data_iterator(x, y, n_epochs, batch_size):
        cost, grads = cost_func(params, x_, y_)
        if np.abs(cost - cost_) < tol:
            break
        params -= lr * grads
        cost_ = cost

        if callback is not None:
            callback(params)

        if verbose:
            print 'epoch: {2}, iter: {0},\tcost: {1}'.format(iter_counter, cost_, i_epoch)
        if display:
            pass

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

class IterLinePlotter(object):
    def __init__(self):
        fig, ax = pl.subplots(1,1)
        self.fig_ = fig
        self.ax_ = ax
        self.bank_ = {}
        self.count_ = 0

    def update(self, id_, xdata, ydata):
        if id_ not in self.bank_:
            line, = self.ax_.plot([], [])
            self.bank_[id_] = line
        line = self.bank_[id_]
        line.set_xdata(np.append(line.get_xdata(), xdata))
        line.set_ydata(np.append(line.get_ydata(), ydata))
        pl.pause(0.0001)
        self.ax_.relim()
        self.ax_.autoscale_view()
        self.fig_.canvas.draw()
            
def stack_image_grid(data):
    n, h, w = data.shape
    r = int(np.round(np.sqrt(n)))
    c = int(np.ceil(n / float(r)))
    res = np.zeros((r*h, c*w))
    grid = np.ogrid[:h, :w]
    for i in range(r):
        for j in range(c):
            k = i*c+j
            if k >= n:
                break
            res[i*h+grid[0], j*w+grid[1]] = data[k]
    return res

def plot_per_iteration():
    shape_ = [self.w0_.shape[0], int(np.sqrt(self.w0_.shape[1])), int(np.sqrt(self.w0_.shape[1]))]
    tmp = stack_image_grid(self.w0_.reshape(shape_))
    im = pl.imshow(tmp, cmap='gray', interpolation='gaussian')
    def plot_(p):
        tmp = shape_[0] * shape_[1] * shape_[2]
        tmp = p[:tmp].reshape(shape_)
        tmp = stack_image_grid(tmp)
        im.set_data(tmp)
        pl.pause(0.001)
        pl.draw()
