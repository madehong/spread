# -*-coding: utf-8-*-
import theano
import numpy as np

np.random.seed(12345)

def normal_weight(shape, loc=0.0, scale=0.1):
    #initialize the weight by normal distribution
    w = np.random.normal(loc=loc, scale=scale, size=shape)
    return w.astype(theano.config.floatX)

def uniform_weight(shape, low=-0.1, high=0.1):
    #initialize the weight by uniform distribution
    w = np.random.uniform(low=low, high=high, size=shape)
    return w.astype(theano.config.floatX)

def ortho_weight(shape):
    #initialize the weight by svd
    assert shape[0] == shape[1], 'ortho_weight\'s must be square!'
    w = np.random.randn(shape[0],shape[0])
    u, s, v = np.linalg.svd(w)
    return u.astype(theano.config.floatX)

def init_bias(shape):
    #initialize the bias by 0
    w = np.zeros(shape)
    return w.astype(theano.config.floatX)

def init_b(ndim):
    #initialize the bias by 0
    w = np.zeros((ndim,))
    return w.astype(theano.config.floatX)
    
def to_shared(value, name):
    #transform value to theano variable
    return theano.shared(value=value, name=name, borrow=True)
