import numpy as np
from scipy import optimize
import pylab as pl
from PIL import Image

random_generator = {'uniform': np.random.uniform}

def init_weights(shape, eps=None, random_type='uniform'):
    '''Xavier initialization. '''

    m, n = shape
    if eps is None:
        eps = 4 * np.sqrt(6.0 / (m + n))
    return random_generator[random_type](low=-eps, high=eps, size=shape)

def register(dic, name):
    def decorator(obj):
        dic[name] = obj
        return obj
    return decorator

def tile_images(data, image_shape, tile_shape, tile_spacing=(1,1)):
    '''Tile up 2-d arrays into one single 2-d array which can be 
       displayed as an image. '''

    data.shape = (-1, image_shape[0], image_shape[1])
    height = tile_shape[0] * image_shape[0] + \
                  (tile_shape[0] - 1) * tile_spacing[0]
    width = tile_shape[1] * image_shape[1] + \
                (tile_shape[1] - 1) * tile_spacing[1]
    tiled = np.zeros((height, width), dtype='uint8')
    k = 0
    for i in range(tile_shape[0]):
        top = i*(image_shape[0]+tile_spacing[0])
        bot = top + image_shape[0]
        for j in range(tile_shape[1]):
            left = j*(image_shape[1]+tile_spacing[1])
            right = left + image_shape[1]
            tmp = data[k].copy()
            tmp -= tmp.min()
            tmp *= 255.0 / (tmp.max() + 1e-8)
            tiled[top:bot, left:right] = tmp.astype('uint8')
            k += 1
    return tiled

def save_image(image, f):
    with open(f, 'wb') as fp:
        tmp = Image.fromarray(image)
        tmp.save(fp)
