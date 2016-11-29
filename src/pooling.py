#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

def softmask(x,mask):
    y = T.exp(x)
    y =y *mask
    sumx = T.sum(y,axis=1)
    x = y/sumx.dimshuffle(0,'x')
    return x

class LastPoolLayer(object):
    def __init__(self, input):
        self.input = input
        self.output = input[-1]
        self.params = []

    def save(self, prefix):
        pass

class MeanPoolLayer(object):
    def __init__(self, input, ll):
        self.input = input
        self.output = T.sum(input, axis=0, acc_dtype='float32') / ll.dimshuffle(0, 'x')          
        self.params = []

    def save(self, prefix):
        pass


class MaxPoolLayer(object):
    def __init__(self, input):
        self.input = input
        self.output = T.max(input, axis = 0)
        self.params = []

    def save(self, prefix):
        pass


        
class TextSimptifiedAttentionLayer(object):
    def __init__(self, rng, input, input_u, input_p, mask, n_wordin, n_usrin, n_prdin, n_out, name, prefix=None):
        self.input = input
        self.inputu = input_u
        self.inputp = input_p

        if prefix is None:
            W_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_wordin + n_out)),
                    high=numpy.sqrt(6. / (n_wordin + n_out)),
                    size=(n_wordin, n_out)
                ),
                dtype=numpy.float32
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

            '''
            v_values = numpy.zeros((n_out,), dtype=theano.config.floatX)            
            v = theano.shared(value=v_values, name='v', borrow=True)
            '''
            v_values = numpy.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=numpy.float32
            )
            v = theano.shared(value=v_values, name='v', borrow=True)
            
            Wu_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_usrin + n_out)),
                    high=numpy.sqrt(6. / (n_usrin + n_out)),
                    size=(n_usrin, n_out)
                ),
                dtype=numpy.float32
            )
            Wu = theano.shared(value=Wu_values, name='Wu', borrow=True)
            
            Wp_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_prdin + n_out)),
                    high=numpy.sqrt(6. / (n_prdin + n_out)),
                    size=(n_prdin, n_out)
                ),
                dtype=numpy.float32
            )
            Wp = theano.shared(value=Wp_values, name='Wp', borrow=True)
            
            u_b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            u_b = theano.shared(value=u_b_values, name='b', borrow=True)

            p_b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            p_b = theano.shared(value=p_b_values, name='b', borrow=True)

            v_b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            v_b = theano.shared(value=v_b_values, name='b', borrow=True)
 
        else:
            print('loading Text simpled params...')
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            v = cPickle.load(f)
            Wu = cPickle.load(f)
            Wp = cPickle.load(f)
            u_b = cPickle.load(f)
            p_b = cPickle.load(f)
            v_b = cPickle.load(f)
            f.close()

        self.W = W
        self.v = v
        self.Wu = Wu
        self.Wp = Wp
        self.u_b = u_b
        self.p_b = p_b
        self.v_b = v_b

        attenu = T.dot(input_u, self.Wu)                                              
        attenp = T.dot(input_p, self.Wp)                                              

        u_atten = T.tanh(T.dot(input, self.W) + attenu + u_b)                         
        u_atten = T.sum(u_atten * v, axis=2, acc_dtype='float32')                 
        u_atten = softmask(u_atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1, 0)        
        u_output = u_atten.dimshuffle(0, 1, 'x') * input
        u_output = T.sum(u_output, axis=0, keepdims=True, acc_dtype='float32')                


        p_atten = T.tanh(T.dot(input, self.W) + attenp + p_b)                         
        p_atten = T.sum(p_atten * v, axis=2, acc_dtype='float32')                 
        p_atten = softmask(p_atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1, 0)        
        p_output = p_atten.dimshuffle(0, 1, 'x') * input
        p_output = T.sum(p_output, axis=0, keepdims=True, acc_dtype='float32')                


        v_atten = T.tanh(T.dot(input, self.W) + attenu + attenp + v_b)                         
        v_atten = T.sum(v_atten * v, axis=2, acc_dtype='float32')                 
        v_atten = softmask(v_atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1, 0)        
        v_output = v_atten.dimshuffle(0, 1, 'x') * input
        v_output = T.sum(v_output, axis=0, keepdims=True, acc_dtype='float32')                

        self.output = T.concatenate([u_output, v_output, p_output], 0)             

        self.params = [self.W, self.v, self.Wu, self.Wp, self.u_b, self.p_b, self.v_b]
        self.name = name
        self.atten = u_atten, p_atten, v_atten
        self.mask = mask

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


class AttentionLayer(object):
    def __init__(self, rng, input, input_u, mask, n_wordin, n_usrin, n_prdin, n_out, name, prefix=None, keepdims=False):
        self.input = input
        self.inputu = input_u
        #self.inputp = input_p

        if prefix is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_wordin + n_out)),
                    high=numpy.sqrt(6. / (n_wordin + n_out)),
                    size=(n_wordin, n_out)
                ),
                dtype=numpy.float32
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

            '''
            v_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            v = theano.shared(value=v_values, name='v', borrow=True)
            '''
            v_values = numpy.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=numpy.float32
            )
            v = theano.shared(value=v_values, name='v', borrow=True)

            Wu_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_usrin + n_out)),
                    high=numpy.sqrt(6. / (n_usrin + n_out)),
                    size=(n_usrin, n_out)
                ),
                dtype=numpy.float32
            )
            Wu = theano.shared(value=Wu_values, name='Wu', borrow=True)
            '''
            Wp_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_prdin + n_out)),
                    high=numpy.sqrt(6. / (n_prdin + n_out)),
                    size=(n_prdin, n_out)
                ),
                dtype=numpy.float32
            )
            Wp = theano.shared(value=Wp_values, name='Wp', borrow=True)
            '''
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        else:
            print('loading attn params...')
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            v = cPickle.load(f)
            Wu = cPickle.load(f)
            #Wp = cPickle.load(f)
            b = cPickle.load(f)
            f.close()

        self.W = W
        self.v = v
        self.Wu = Wu
        #self.Wp = Wp
        self.b = b

        attenu = T.dot(input_u, self.Wu)
        #attenp = T.dot(input_p, self.Wp)

        atten = T.dot(input, self.W)
        atten = atten + attenu
        #atten = atten + attenp
        atten = atten + b
        #atten = T.tanh(T.dot(input, self.W)+ attenu + attenp +b)
        atten = T.sum(atten * v, axis=2, acc_dtype='float32')
        atten = softmask(atten.dimshuffle(1,0), mask.dimshuffle(1,0)).dimshuffle(1, 0)
        output = atten.dimshuffle(0, 1, 'x') * input
        self.output = T.sum(output, axis=0, acc_dtype='float32',keepdims=keepdims)

        #self.params = [self.W, self.v,self.Wu,self.Wp,self.b]
        self.params = [self.W, self.v, self.Wu, self.b]
        self.name = name
        self.atten = atten
        self.mask = mask

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()



class AttentionLayer1(object):
    def __init__(self, rng, input, input_target, input_category, mask, n_wordin, n_targetin, n_categoryin, n_out, name, prefix=None, keepdims=False):
        self.input = input
        self.inputu = input_target

        if prefix is None:
            W_values = numpy.asarray(                                              
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_wordin + n_out)),
                    high=numpy.sqrt(6. / (n_wordin + n_out)),
                    size=(n_wordin, n_out)
                ),
                dtype=numpy.float32
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

            '''
            v_values = numpy.zeros((n_out,), dtype=theano.config.floatX)            
            v = theano.shared(value=v_values, name='v', borrow=True)
            '''
            v_values = numpy.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=numpy.float32
            )
            v = theano.shared(value=v_values, name='v', borrow=True)
            
            W_target_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_targetin + n_out)),
                    high=numpy.sqrt(6. / (n_targetin + n_out)),
                    size=(n_targetin, n_out)
                ),
                dtype=numpy.float32
            )
            W_target = theano.shared(value=W_target_values, name='Wu', borrow=True)

            W_category_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_categoryin + n_out)),
                    high=numpy.sqrt(6. / (n_categoryin + n_out)),
                    size=(n_categoryin, n_out)
                ),
                dtype=numpy.float32
            )
            W_category = theano.shared(value=W_category_values, name='Wu', borrow=True)

            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
 
        else:
            print('loading attn params...')
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            v = cPickle.load(f)
            W_target = cPickle.load(f)
            W_category = cPickle.load(f)
            b = cPickle.load(f)
            f.close()

        self.W = W
        self.v = v
        self.W_target = W_target
        self.W_category = W_category
        self.b = b

        atten_target = T.dot(input_target, self.W_target)
        atten_category = T.dot(input_category, self.W_category)

        atten = T.tanh(T.dot(input, self.W) + atten_target + atten_category + b)
        atten = T.sum(atten * v, axis=2, acc_dtype='float32')                 
        atten = softmask(atten.dimshuffle(1, 0), mask.dimshuffle(1, 0)).dimshuffle(1, 0)
        output = atten.dimshuffle(0, 1, 'x') * input
        self.output = T.sum(output, axis=0, acc_dtype='float32',keepdims=keepdims)                

        self.params = [self.W, self.v, self.W_target, self.b]
        self.name = name
        self.atten = atten
        self.mask = mask

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


class Dropout(object):
    def __init__(self, input, rate, istrain):
        rate = numpy.float32(rate)
        self.input = input
        srng = T.shared_randomstreams.RandomStreams()
        mask = srng.binomial(n=1, p=numpy.float32(1-rate), size=input.shape, dtype='float32')
        self.output = T.switch(istrain, mask*self.input, self.input*numpy.float32(1-rate))
        self.params = []

    def save(self, prefix):
        pass
