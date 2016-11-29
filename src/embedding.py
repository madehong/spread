# -*- coding: utf-8 -*-
from superclass import SuperLayer
from utils import normal_weight, init_bias, to_shared

class Embedding(SuperLayer):
    def __init__(self, name, x, n_voc, ndim, init_value=None, path=None, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)
        if path is None:
            if init_value == None:
                self.params['E'] = to_shared(init_func((n_voc, ndim)), self.name + '_E')
            else:
                self.params['E'] = to_shared(init_value, self.name + '_E')
        self.output = self.stream(x)

    def stream(self, x):
        return self.params['E'][x]
