"""
Extract from datasets or generate synthetic data, determine data type, and
divide to train, test (validation?); pickle processed data for repeated use

datasets: 
http://scikit-learn.org/stable/datasets/#rcv1-dataset
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html#sklearn.datasets.fetch_rcv1
https://www.cs.toronto.edu/~kriz/cifar.html

Input data type: ndarray (dense), sparse, list, None

Output data type: ndarray

April-2016 by Cheng Tang (chengtang48@gmail.com)
"""
import scipy.sparse as sp
import numpy as np
import random
import pdb

import logging
import os
from six.moves import cPickle as pickle

from sklearn.datasets import fetch_rcv1, fetch_mldata, fetch_covtype
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import as_float_array
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn.decomposition import RandomizedPCA

## local modules
from image_preprocess import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
                    
data_bank = {'rcv1','mnist','gauss',
			 'covtype','cifar10_white_norm','cifar10_norm',
			 'cifar10_raw'
           }

def maybe_pickle(dataname, force = False, verbose = True):
	filename = dataname + '.pickle'
	"""Process and pickle a dataset if not present"""
	if force or not os.path.exists(filename):
		train, test, meta = get_data(dataname, verbose = verbose)
		
		# pickle the dataset
		print('Pickling train, test, meta info to file %s' % filename)
		try:
			with open(filename, 'wb') as f:
				pickle.dump([train, test, meta], f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save to', filename, ':', e)	
	else:
		print('%s already present - Skipping pickling.' % filename)
		with open(filename, 'rb') as f:
			train, test, meta = pickle.load(f)
		
	return train, test, meta

def get_data(dataname, verbose = True):

	assert dataname in data_bank, 'Dataset name not recognized!'
	
	# meta information about the data
	meta_dict = {'isSparse':False, 'n_true_classes':None, 
	  'xsn_train':None,'xsn_test':None} 
	
	train_lb = None
	if dataname == 'rcv1':
		print('Fetching RCV1 data from sklearn')
		train = fetch_rcv1(subset='test')
		train = train.data
		test = fetch_rcv1(subset='train')
		test = test.data
		meta_dict['n_true_classes'] = 103

	elif dataname == 'mnist':
		print('Fetching MNIST data from sklearn')
		mnist = fetch_mldata('MNIST original')
		data_ind = range(mnist.data.shape[0])
		random.shuffle(data_ind)
		train_ind = data_ind[:60000]
		test_ind = data_ind[-10000:]
		
		train = mnist.data[train_ind,:]
		test = mnist.data[test_ind,:]
		meta_dict['n_true_classes'] = 10
		
	elif dataname == 'gauss':
		""" Synthetic Gaussian """
		print('Generating Gaussian blurbs datasets')
		#centers = [[2, 2], [-2, -2], [2, -2]]
		#centers = np.asarray(centers)
		n_s = 7000
		gauss, _ = make_blobs(n_samples=n_s, n_features = 10,
		   centers=50, center_box = (-30,30), cluster_std=10.0)
		
		data_ind = range(gauss.shape[0])
		random.shuffle(data_ind)
		train_ind = data_ind[:6*n_s/7]
		test_ind = data_ind[-n_s/7:]
		
		train = gauss[train_ind,:]
		test = gauss[test_ind,:]
		meta_dict['n_true_classes'] = 50
	
	elif dataname == 'covtype':
		""" Forest covertype """
		print('Fetching forest covertype datasets')	
		cov = fetch_covtype()
		train = cov.data[:500000]
		train_lb = cov.target[:500000]
		test = cov.data[500000:]
		test_lb = cov.target[500000:]
		meta_dict['n_true_classes'] = 7
		
	elif dataname == 'cifar10_raw':
		"""
		This will make over 4 million data points for training, each with
		dim 8*8*3
		"""
		# take user input to know which batch to preprocess
		usr_input = raw_input('Which batch to preprocess: ')
		# get data home
		data_home = '/Users/tangch/scikit_learn_data/cifar10' # make this portable in the future
		
		os.chdir(data_home)
		
		prefix = 'data_batch_'
		# combine training batches
		if usr_input != 'test':
			curr_batch = prefix + str(usr_input)
			fname = curr_batch
			dataname = dataname + 'batch_' + str(usr_input)
		else:
			fname = data_home + 'test_batch'
			dataname = dataname + 'test_batch' 
			#print 'Opening '+fname + 'in directory' + os.getcwd()
		with open(fname, 'rb') as f:
			dict = pickle.load(f)

		train = dict['data']
		train_lb = dict['labels']
		
		print 'processing batch %s' % fname
				
		print '%d by %d training data with %d label' %(train.shape[0],\
		  train.shape[1], len(train_lb))
		os.chdir('/Users/tangch/Documents/Python_projects/myprojects/mbkm_2016/mbkm')
		
		# Reduce the dimension of data by random sampling
		train = train.reshape(train.shape[0],32,32,3)
		width = 8
		height = width
		
		# subsampling patches u.a.r.
		train, train_lb = convsubsample(train, 1, width,height, labels = train_lb)
		# random shuffle training data
		#ind = range(train.shape[0])
		print 'shuffling dataset randomly'
		#ind = random.shuffle(ind)
		#pdb.set_trace()
		#train = train[ind]
		#train_lb = train_lb[ind]
		train = np.random.sample(train, train.shape[0])
		train = np.array(train)
		train_lb = np.random.sample(train_lb, train_lb.shape[0])
		train_lb = np.array(train_lb)
		test = None
			
		# get test data/labels
		#fname = data_home + 'test_batch'
		#with open(fname,'rb') as f:
		#	dict = pickle.load(f)
		#test = dict['data']
		#test_lb = dict['labels']
		
		# subsampling patches u.a.r.
		#test, test_lb = convsubsample(test, 1, width,height, labels = test_lb)
		
		meta_dict['n_true_classes'] = 10
		print 'finished preprocessing current batch'
	
	elif dataname == 'cifar10_norm':
		""" 
		  Only works if we have cifar10_raw
		"""
		usr_input = raw_input('Which batch to preprocess: ')
		### load cifar10_raw
		fname = 'cifar10_raw'
		if usr_input != 'test':
			fname = fname + 'batch_' + str(usr_input)
			dataname = dataname + 'batch_' + str(usr_input)
		else:
			fname = fname + 'test_batch'
			dataname = dataname + 'test_batch'
		
		try:
			with open(fname, 'rb') as f:
				train, test, meta = pickle.load(f)
		except Exception as e:
			print('Cannot open file '+fname)
			
		### Normalization 
		train = normalize(train, norm = 'l2')
		test = normalize(test, norm = 'l2')
		
		### meta info extraction
		meta_dict['n_true_classes'] = meta['n_true_classes']
		train_lb = meta['train_lb']
		test_lb = meta['test_lb']
		
		
	elif dataname == 'cifar10_white_norm':
		""" 
		  Only works if we have cifar10_norm
		"""
		### load cifar10_norm
		fname = 'cifar10_norm'
		try:
			with open(fname, 'rb') as f:
				train_old, test_old, meta = pickle.load(f)		
		except Exception as e:
			print('Cannot find file '+fname)
		
		train_test = np.vstack((train_old, test_old))
		### Whitening
		pca = RandomizedPCA(whiten = True) #use approx PCA to save computation
		train_test = pca.fit_transform(train_test)
		train = train_test[:train_old.shape[0]]
		test = train_test[train_old.shape[0]:]
		### Extract meta info
		meta_dict['n_true_classes'] = meta['n_true_classes']
		train_lb = meta['train_lb']
		test_lb = meta['test_lb']
		
	else:
		print 'nothing'
	
	meta_dict['dataname'] = dataname
	
	# add true labels if exists
	if not train_lb is None:
		meta_dict['train_lb'] = train_lb
		#meta_dict['test_lb'] = test_lb
		
	# Check if data is sparse
	if sp.issparse(train):
			meta_dict['isSparse'] = True
			print ('The %s data is sparse' % dataname)
					
	print('%d training data' % train.shape[0])
	print ('data dimension is %d' % train.shape[1])
	if len(train.shape) == 3:
		print ('data has %d channels' % train.shape[2])
	
	if test is None:
		print 'No test data'
	else:
		print('%d test data' % test.shape[0])
	print('The number of true classes is %d' % meta_dict['n_true_classes'])
	
	# What does this do?
	train = as_float_array(train, copy=True)
	if not test is None:
		test = as_float_array(test,copy=True)
	
	# precompute squared norms for faster computation
	x_squared_norms_tr = row_norms(train, squared=True) 
	meta_dict['xsn_train'] = x_squared_norms_tr
	if not test is None:
		x_squared_norms_tt = row_norms(test,squared=True)
		meta_dict['xsn_test'] = x_squared_norms_tt
	
	return train, test, meta_dict

# Test
if __name__ == '__main__':
	maybe_pickle('rcv1')
	maybe_pickle('mnist')
	maybe_pickle('gauss')
	maybe_pickle('covtype')
	maybe_pickle('cifar10_raw')
		