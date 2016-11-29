# -*- coding:utf-8 -*-
import numpy as np
np.random.seed(12345)

def preprocess(word_dict, label_dict, path):
	x = []
	y = []
	f = open(path)
	label_inx = len(label_dict)
	word_inx = len(word_dict)
	for line in f:
		line = line.strip()
		words = line.split()

		label = words[0].split(':')[0]
		if label not in label_dict:
			label_dict[label] = label_inx
			label_inx += 1
		label = label_dict[label]
		
		word_list = words[1:]
		tmp = []
		for word in word_list:
			if word not in word_dict:
				word_dict[word] = word_inx
				word_inx += 1
			tmp.append(word_dict[word])
		x.append(tmp)
		y.append(label)
	f.close()
	return zip(x,y), word_dict, label_dict

def padding(data, maxlen=None):
	if maxlen == None:
		seq_len = map(lambda x:len(x), data)
		maxlen = max(seq_len)
	X = []
	M = []
	for x_ in data:
		x_ = x_[:maxlen] + [0 for i in xrange(maxlen - len(x_))]
		m_ = [1 for i in xrange(len(x_[:maxlen]))] + [0 for i in xrange(maxlen - len(x_))]
		X.append(x_)
		M.append(m_)
	return X, M

class DataSet(object):
	def __init__(self, train_path='./trec-train.txt', test_path='./trec-test.txt', maxlen=50, batch_size=50, ndim=200,valid_split=0.1):
		
		self.maxlen = maxlen
		self.batch_size = batch_size
		self.ndim = ndim

		self.label_dict = {}
		self.word_dict  = {}

		self.train_set, self.word_dict, self.label_dict = preprocess(self.word_dict, self.label_dict, train_path)
		self.test_set , self.word_dict, self.label_dict = preprocess(self.word_dict, self.label_dict, test_path)

                self.valid_set = self.train_set[int(len(self.train_set)*(1 - valid_split)):]
                self.train_set = self.train_set[:int(len(self.train_set)*(1 - valid_split))]

		self.train_epoch = len(self.train_set) // self.batch_size
		if len(self.train_set) % self.batch_size:
			self.train_epoch += 1

		self.valid_epoch = len(self.valid_set) // self.batch_size
		if len(self.valid_set) % self.batch_size:
			self.valid_epoch += 1

		self.test_epoch  = len(self.test_set)  // self.batch_size
		if len(self.test_set)  % self.batch_size:
			self.test_epoch  += 1

		self.test_batches = self.get_batches(self.test_set, is_shuffle=False)
		self.valid_batches= self.get_batches(self.valid_set,is_shuffle=False)
		self.train_batches= self.get_batches(self.train_set,is_shuffle=True)

                self.n_voc = len(self.word_dict)
                self.n_lab = len(self.label_dict)

	def get_batches(self, data, is_shuffle=True):

		if is_shuffle:
			np.random.shuffle(data)
		X = []
		Y = []
		for x, y in data:
			X.append(x)
			Y.append(y)

		batches = []

		epoch = len(X) // self.batch_size
		if len(X) % self.batch_size:
			epoch += 1
		def to_int(x):
			return np.asarray(x).astype('int32') 
		def to_float(x):
			return np.asarray(x).astype('float32')
		for i in xrange(epoch):
			tmp = X[i*self.batch_size:(i+1)*self.batch_size]
			y   = Y[i*self.batch_size:(i+1)*self.batch_size]
			x, m= padding(tmp)
			batches.append((to_int(x).T,to_float(m).T,to_int(y)))

		return batches

if __name__ == '__main__':

	data = DataSet('./trec-train.txt', './trec-test.txt',)
