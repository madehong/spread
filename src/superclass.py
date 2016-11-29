# -*- coding: utf-8 -*-
import cPickle
from updates import *
from collections import OrderedDict
import numpy

class SuperLayer(object):
    def __init__(self, name, path):
        self.name = name

        if path:
            self.load()
        else:
            self.params = OrderedDict()

    def stream(self):
        # Your operations on params and inputs
        # this method must be overwritten in subclass
        pass

    def load(self, path):
        with open(path + self.name + '.params', 'rb') as f:
            self.params = cPickle.load(f)

    def dump(self, path):
        with open(path + self.name + '.params', 'wb') as f:
            cPickle.dump(self.params, f)
'''
class SuperModel(object):
    def __init__(self, name):
        self.name = name
        self.cost = 0.
        self.layers = []
    def train_model(self):
        #Overwrite this method that includes you training codes in subclass!
        pass
    def valid_model(self):
        #Overwrite this method that includes you validing codes in subclass!
        pass
    def test_model(self):
        #Overwrite this method that includes you testing  codes in subclass!
        pass
    def get_updates(self, l1_norm=0., l2_norm=1e-5, optimizer=adagrad):
        params = []
        for layer in self.layers:
            params += list(layer.params.values())
        l1_norm = numpy.float32(l1_norm)
        l2_norm = numpy.float32(l2_norm)
        for param in params:
            if l1_norm > 0.:
                self.cost += T.sum(l1_norm * T.abs_(param), acc_dtype='float32')
            if l2_norm > 0.:
                self.cost += T.sum(l2_norm * T.sqrt(param), acc_dtype='float32')
        gparams = [T.grad(self.cost, param) for param in params]
        updates = adagrad(params, gparams)
        return updates
'''
