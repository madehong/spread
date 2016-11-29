# -*- coding: utf-8 -*-
from theano import tensor as T
import numpy as np
from superclass import SuperLayer

class Dropout(SuperLayer):
    def __init__(self, name, x, rate, is_train, path=None):
        SuperLayer.__init__(self, name, None)
        self.output = self.stream(x, rate, is_train)

    def stream(self, x, rate, is_train):
        srng = T.shared_randomstreams.RandomStreams()
        mask = srng.binomial(n=1, p=np.float32(1-rate), size=x.shape, dtype='float32')
        return T.switch(is_train, mask*x, x*np.float32(1-rate))
