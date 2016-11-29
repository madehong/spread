# -*- coding: utf-8 -*-
import theano
from theano import tensor as T
import numpy as np

from superclass import SuperLayer
from utils import normal_weight, init_bias, to_shared

class LSTM(SuperLayer):
    def __init__(self, name, x, mask, n_in, hidden_size, n_out, path=None, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)
        if path is None:
            self.params['Wi'] = to_shared(init_func((n_in, hidden_size)), self.name + '_Wi')
            self.params['Wf'] = to_shared(init_func((n_in, hidden_size)), self.name + '_Wf')
            self.params['Wo'] = to_shared(init_func((n_in, hidden_size)), self.name + '_Wo')
            self.params['Wc'] = to_shared(init_func((n_in, hidden_size)), self.name + '_Wc')

            self.params['Ui'] = to_shared(init_func((hidden_size, hidden_size)), self.name + '_Ui')
            self.params['Uf'] = to_shared(init_func((hidden_size, hidden_size)), self.name + '_Uf')
            self.params['Uo'] = to_shared(init_func((hidden_size, hidden_size)), self.name + '_Uo')
            self.params['Uc'] = to_shared(init_func((hidden_size, hidden_size)), self.name + '_Uc')
            
            self.params['bi'] = to_shared(init_bias((n_out,)), self.name + '_bi')
            self.params['bf'] = to_shared(init_bias((n_out,)), self.name + '_bf')
            self.params['bo'] = to_shared(init_bias((n_out,)), self.name + '_bo')
            self.params['bc'] = to_shared(init_bias((n_out,)), self.name + '_bc')
        self.output = self.stream(x, mask)

    def stream(self, x, mask):
        def step(x, mask, h_, c_):
            i = T.nnet.sigmoid(T.dot(x, self.params['Wi']) + T.dot(h_, self.params['Ui']) + self.params['bi'])
            f = T.nnet.sigmoid(T.dot(x, self.params['Wf']) + T.dot(h_, self.params['Uf']) + self.params['bf'])
            o = T.nnet.sigmoid(T.dot(x, self.params['Wo']) + T.dot(h_, self.params['Uo']) + self.params['bo'])
            C =         T.tanh(T.dot(x, self.params['Wc']) + T.dot(h_, self.params['Uc']) + self.params['bc'])

            C_ = c_ * f + C * i
            C_ = C_ * mask.dimshuffle(0,'x') 
            C_ = T.cast(C_,'float32')
            H_ = T.tanh(C_) * o
            H_ = H_ * mask.dimshuffle(0,'x') 
            H_ = T.cast(H_,'float32')
            return [H_, C_]

        outputs, _ = theano.scan(fn=step,
                                 outputs_info=[T.zeros_like(T.dot(x[0], self.params['Wi'])), T.zeros_like(T.dot(x[0], self.params['Wi']))],
                                 sequences=[x, mask])
        return outputs[0]
