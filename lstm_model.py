#-*- coding: UTF-8 -*-
from src.updates import adagrad
from src.embedding import Embedding
from src.rnn import LSTM
from src.attention import Attention
from src.dense import Dense
from src.dropout import Dropout
import theano
import theano.tensor as T
import sys

class Model():
    def __init__(self, name, data, path=None):
        
        self.name = name 
        self.data = data

        n_voc = data.n_voc
        n_lab = data.n_lab
        ndim = data.ndim
        
        #build model
        x = T.imatrix('x')
        m = T.fmatrix('mask')
        y = T.ivector('y')
        is_train = T.iscalar('train_flag')

        self.layers = []
        self.layers.append(Embedding('embedding', x, n_voc, ndim, path))
        self.layers.append(LSTM('lstm', self.layers[-1].output, m, ndim, ndim, ndim, path))
        self.layers.append(Attention('attention', self.layers[-1].output, T.mean(self.layers[-1].output, 0),  m, ndim, ndim, ndim, path))
        self.layers.append(Dense('full_connection', self.layers[-1].output, ndim, ndim, path))
        self.layers.append(Dropout('dropout', self.layers[-1].output, 0.5, is_train, path))
        self.layers.append(Dense('softmax', self.layers[-1].output, ndim, int(n_lab), path, activation=T.nnet.softmax))
        
        #define cost function
        self.cost = -T.mean(T.log(self.layers[-1].output)[T.arange(y.shape[0]), y], acc_dtype='float32')
        correct = T.sum(T.eq(T.argmax(self.layers[-1].output, axis=1), y), acc_dtype='int32')
        
        #get grads of params
        params = []
        for layer in self.layers:
            params += list(layer.params.values())
        gparams = T.grad(self.cost, wrt=params)
        updates = adagrad(params, gparams)

        #define training model and test model
        self.train_model = theano.function(
            inputs=[is_train, x, m, y],
            outputs=self.cost,
            updates=updates)

        self.acc_model = theano.function(
            inputs=[is_train, x, m, y],
            outputs=[correct])
        self.index = {}
        self.index['valid'] = 1
        self.index['test']= 1
        self.best_valid_acc = 0.0
        self.out_len = 0
    
    #train function
    def train(self, result_file, valid_frequency=20):
        n = 0
        valid_acc = 0.
        for i in xrange(self.data.train_epoch):
            n += 1
            out = self.train_model(1,
                                   self.data.train_batches[i][0],
                                   self.data.train_batches[i][1],
                                   self.data.train_batches[i][2])
            if n % valid_frequency == 0:
                vacc = self.accuracy('valid', self.data.valid_batches, result_file)
                if vacc > self.best_valid_acc:
                    tacc = self.accuracy('test', self.data.test_batches, result_file)
                    out = '##### Best Valid Acc: ' + str(self.best_valid_acc) + '; With Test Acc: ' + str(tacc)
                    sys.stdout.write('\b'*self.out_len + out)
                    self.out_len = len(out)
                    sys.stdout.flush()
                    self.best_valid_acc = vacc
        self.data.train_batches = self.data.get_batches(self.data.train_set, is_shuffle=True)

    #accuracy function
    def accuracy(self, which, data, result_file, record=False):
        cor = 0
        tot = 0
        for i in xrange(len(data)):
            tmp = self.acc_model(0,
                                 data[i][0],
                                 data[i][1],
                                 data[i][2])
            cor += tmp[0]
            tot += len(data[i][2])
        acc = float(cor)/float(tot)
        if record:
            f = open(result_file, 'a')
            f.write(which + str(self.index[which]) + ': ACC: ' + str(acc) + '\n')
            self.index[which] += 1
            f.flush()
            f.close()
        return acc
