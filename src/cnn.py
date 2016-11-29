# -*- coding: utf-8 -*-
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from superclass import SuperLayer
from utils import normal_weight, init_bias, to_shared

class CNNPool(SuperLayer):
    def __init__(self, name, image, filter_shape, image_shape, pooling_size, path=None, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pooling_size = pooling_size

        if path is None:
            self.params['W'] = to_shared(init_func(filter_shape), self.name + '_W')
            self.params['b'] = to_shared(init_bias((filter_shape[0],)), self.name + '_b')

        self.output = self.stream(image)

    def stream(self, image):
        conv2d_output = conv2d(input=image,
                               filters=self.params['W'],
                               filter_shape=self.filter_shape,
                               input_shape=self.image_shape)

        pool_output = pool.pool_2d(input=conv2d_output,
                                   ds=self.pooling_size,
                                   ignore_border=True)
        return T.tanh(pool_output + self.params['b'].dimshuffle('x', 0, 'x', 'x'))


class CNN(SuperLayer):
    def __init__(self, name, image, filter_shape, image_shape, path=None, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)

        self.filter_shape = filter_shape
        self.image_shape = image_shape

        if path is None:
            self.params['W'] = to_shared(init_func(filter_shape), self.name + '_W')
            self.params['b'] = to_shared(init_bias((filter_shape[0],)), self.name + '_b')

        self.output = self.stream(image)

    def stream(self, image):
        conv2d_output = conv2d(input=image,
                               filters=self.params['W'],
                               filter_shape=self.filter_shape,
                               input_shape=self.image_shape)

        return T.tanh(conv2d_output + self.params['b'].dimshuffle('x', 0, 'x', 'x'))


class Pooling(SuperLayer):
    def __init__(self, name, x, pooling_size, path=None):
        SuperLayer.__init__(self, name, path)
        self.pooling_size = pooling_size
        self.output = self.stream(x)

    def stream(self, x):
        pool_output = pool.pool_2d(input=x,
                                  ds=self.pooling_size,
                                  ignore_border=True)
        return pool_output
