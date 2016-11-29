# -*- coding: utf-8 -*-
from theano import tensor as T
from superclass import SuperLayer
from utils import normal_weight, init_bias, to_shared

class Hidden(SuperLayer):
    def __init__(self, name, x, n_in, n_out, path=None, activation=T.tanh, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)
        self.activation = activation
        if path is None:
            self.params['W'] = to_shared(init_func((n_in, n_out)), self.name + '_W')
            self.params['b'] = to_shared(init_bias((n_out,)), self.name + '_b')
        self.output = self.stream(x)

    def stream(self, x):
        return self.activation(T.dot(x, self.params['W']) + self.params['b'])
