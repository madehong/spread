
# -*- coding: UTF-8 -*-
from lstm_model import Model
from trec.dataset import DataSet

if __name__ == '__main__':

    result_file = 'trec_result.txt'

    print 'loading data...'
    data = DataSet('./trec/trec-train.txt', './trec/trec-test.txt')

    print('building model...')
    model = Model('LSTM for trec', data)

    print('training with testing...')
    for i in xrange(1, 400):
        model.train(result_file)
