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