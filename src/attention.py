# -*- coding: utf-8 -*-
import theano
from theano import tensor as T

from superclass import SuperLayer
from utils import normal_weight, init_bias, to_shared

class Attention2(SuperLayer):
    def __init__(self, name, x, attn1, attn2, mask, x_in, attn1_in, attn2_in, n_out, path=None, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)
        if path is None:
            self.params['Wx']   = to_shared(init_func((x_in, n_out)    ), self.name + '_Wx'  )
            self.params['Wv']   = to_shared(init_func((n_out,)         ), self.name + '_Wv'  )
            self.params['Wa_1'] = to_shared(init_func((attn1_in, n_out)), self.name + '_Wa_1')
            self.params['Wa_2'] = to_shared(init_func((attn2_in, n_out)), self.name + '_Wa_2')
            self.params['b']    = to_shared(init_bias((n_out,)         ), self.name + '_b'   )
        self.output = self.stream(x,attn1,attn2,mask)

    def stream(self, x, attn1, attn2, mask):

        attn_x = T.dot(x, self.params['Wx'])                                              
        attn_a1= T.dot(attn1, self.params['Wa_1'])                                              
        attn_a2= T.dot(attn2, self.params['Wa_2'])                                              

        atten = T.tanh(attn_x + attn_a1 + attn_a2 + self.params['b'])                         
        atten = T.sum(atten * self.params['Wv'], axis=2, acc_dtype='float32')                 

        def softmax(x, mask):
            exp_x = T.exp(x)
            exp_x = exp_x * mask
            sum_x = T.sum(exp_x, 1)
            prob = exp_x / sum_x.dimshuffle(0, 'x')
            return prob

        atten = softmax(atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1,0) 
        output = atten.dimshuffle(0, 1, 'x') * x
        return T.sum(output, axis=0, acc_dtype='float32') 

class Attention(SuperLayer):
    def __init__(self, name, x, a, mask, x_in, attn_in, n_out, path=None, init_func=normal_weight):
        SuperLayer.__init__(self, name, path)
        if path is None:
            self.params['Wx'] = to_shared(init_func((x_in, n_out)   ), self.name + '_Wx')
            self.params['Wv'] = to_shared(init_func((n_out,)        ), self.name + '_Wv')
            self.params['Wa'] = to_shared(init_func((attn_in, n_out)), self.name + '_Wa')
            self.params['b']  = to_shared(init_bias((n_out,)        ), self.name + '_b' )
        self.output = self.stream(x,a,mask)

    def stream(self, x, a, mask):

        attn_x = T.dot(x, self.params['Wx'])                                              
        attn_a = T.dot(a, self.params['Wa'])                                              

        atten = T.tanh(attn_x + attn_a + self.params['b'])                         
        atten = T.sum(atten * self.params['Wv'], axis=2, acc_dtype='float32')                 

        def softmax(x, mask):
            exp_x = T.exp(x)
            exp_x = exp_x * mask
            sum_x = T.sum(exp_x, 1)
            prob = exp_x / sum_x.dimshuffle(0, 'x')
            return prob

        atten = softmax(atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1,0) 
        output = atten.dimshuffle(0, 1, 'x') * x
        return T.sum(output, axis=0, acc_dtype='float32') 
