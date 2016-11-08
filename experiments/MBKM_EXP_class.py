# papers:
# Convergence property of the k-means algorithm: http://papers.nips.cc/paper/989-convergence-properties-of-the-k-means-algorithms.pdf 
# Webscale k-means: http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
# Convergence rate of stochastic k-means: https://arxiv.org/abs/1610.04900
#
################### author: Cheng Tang (chengtang48@gmail.com), April. 2016

#import pdb
from time import time
import numpy as np
#import scipy.sparse as sp
import matplotlib.pyplot as plt
import random
import math
import os
from six.moves import cPickle as pickle

#from sklearn.cluster import k_means_
#from sklearn.cluster import _k_means
from sklearn.cluster import MiniBatchKMeans, KMeans
#from sklearn.metrics.pairwise import pairwise_distances_argmin
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn import metrics

###### local modules
from data_prep import maybe_pickle
from seedingAlgs import setSeed
from KMeans_ext import KMeans_ext
from utilities import _compute_inertia
from MBKM_ext import MiniBatchKMeans_ext


class MBKM_EXP(object):
	""" Experiment object for kmeans variants """
	## var_names: 'learn_rate','mb_size','seeding', 'k', 'dataname', 'alg_mode', 'eval_mode'	
	def __init__(self, dataname, n_clusters, n_iters, n_runs, 
	    alg_mode = 'mbkm', seed_method = 'random', 
	    minibatch_size = None, learn_rate = (0,1), verbose = False):
		"""
		Algorithmic variables:
		- alg: currently used clustering algorithm 
		- n_curr_iters: current number of iterations
		- n_curr_runs: currently executed number of runs
		- init_centroids: set of latest seeds
		- centroids: set of current centroids
		- curr_stat: cost vs iters vector of current run

		
		Experiment parameters/variables
		- dataname: name of current choice of dataset
		- train
		- test
		- xsn_train: precomputed row squared norms of training set
		- xsn_test
		- alg_mode: 'mbkm' or 'km'
		- seed_method: current seeding method
		- n_clusters: current number of clusters
		- n_iters: total number of iterations
		- n_runs: total number of runs
		- minibatch_size
		- learn_rate - (c, n)	

		"""
		## Get dataset
		self.train, self.test, meta = maybe_pickle(dataname)
		self.xsn_train = meta['xsn_train']
		self.xsn_test = meta['xsn_test']
		self.dataname = meta['dataname']
		## Set up experiment parameters
		self.alg_mode = alg_mode
		self.seed_method = seed_method
		self.n_clusters = n_clusters
		self.n_iters = n_iters
		self.n_runs = n_runs
		self.minibatch_size = minibatch_size
		self.learn_rate = learn_rate
		self.eval_mode = 'test'
		
		self.verbose = verbose
		
		# Create a variable snapshot
		self.make_var_snapshot()
		
		## Initialize algoirthmic parameters
		self.n_curr_iters = 0
		self.n_curr_runs = 0
		self.centroids = None
		self.curr_stat = np.zeros(n_iters)

#	def seed(self):
#		print('Seeding with %s' % seed_method)
#		self.seeds = setSeed(train, n_clusters, seed_method, 
#         x_squared_norms_train)
#        print('Successfully initialized %d centroids' % self.n_clusters)

	
	def set_eval_mode(self, eval_mode):
		self.eval_mode = eval_mode # evaluation on train or test
		
	def multiple_runs(self):
		'''
		Is this incorrect?
		'''
		# reset run counter
		self.n_curr_runs = 0
		accum_stat = np.zeros([self.n_runs, self.n_iters])
		accum_runtime = np.zeros([self.n_runs])
		for i in range(self.n_runs):
			accum_runtime[self.n_curr_runs] = self.single_run()
			accum_stat[self.n_curr_runs] = self.curr_stat
			self.n_curr_runs += 1
	
		return np.mean(accum_stat, 0), np.std(accum_stat,0), np.mean(accum_runtime)
		
	def single_run(self):
		# Reset iteration counter
		self.n_curr_iters = 0
		## Seeding
		self.init_centroids = setSeed(self.train, self.n_clusters, self.xsn_train,\
		  seed_method = self.seed_method)
		
		if self.verbose and type(self.seed_method) is str:
			print('Successfully initialized %d centroids with %s' % (self.n_clusters,self.seed_method))
		else:
			print('Initialized with self-defined centers')
		## Initialize algorithm
		if self.alg_mode == 'mbkm':
			self.alg = MiniBatchKMeans_ext(n_clusters=self.n_clusters, init=self.init_centroids.copy(),
		      n_init=1,verbose= False, reassignment_ratio = 0)
		else:
			self.alg = KMeans_ext(n_clusters = self.n_clusters, init = self.init_centers.copy(), n_init = 1,
	  		    verbose = False)
		runtime = 0  
		for i in range(self.n_iters):
			runtime = runtime + self.step()
		if self.verbose:
			print('Finished the %d run of %s' %(self.n_curr_runs+1, self.alg_mode))
			print('Killing algorithm instance..')
		
		self.alg = None
		
		return runtime
		
	def step(self):
		if self.alg_mode == 'mbkm':
			t0 = time()
			sample_idx = random.sample(range(self.train.shape[0]), self.minibatch_size)
			self.alg.partial_fit(self.train[sample_idx,:])
			td = time()
			## Reset algorithm variables
			self.centroids = self.alg.cluster_centers_
		else:
			# kmeans
			t0 = time()
			self.alg.partial_fit(self.train)
			td = time()
        	## Reset algorithm variables
			self.centroids = self.alg.cluster_centers_
			
        ## Record to stat
		if self.eval_mode == 'train':
			eval_data = self.train	
			xsn_eval_data = self.xsn_train	
		else:	
			eval_data = self.test	
			xsn_eval_data = self.xsn_test	
			
		self.curr_stat[self.n_curr_iters] = _compute_inertia(eval_data,xsn_eval_data,self.centroids)[1]	
		if self.verbose:
			print('The %s cost at iteration %d is %f' %(self.eval_mode, self.n_curr_iters,\
		      self.curr_stat[self.n_curr_iters]))
		
		self.n_curr_iters += 1
		
		return td-t0	

	def make_var_snapshot(self):
		""" report current var names """
		self.var_snapshot = {'learn_rate': self.learn_rate, 'mb_size': self.minibatch_size,
             'seed_method': self.seed_method, 'k': self.n_clusters, 
             'dataname': self.dataname, 'alg_mode': self.alg_mode, 
             'eval_mode': self.eval_mode
             }
#	def run_exp(self, var_name = None, var_value = None):
#		""" Run experiment based on current variables"""
#		# update variable snapshot
#		self.var_snapshot[var_name] = var_value
#		# run experiment
#		avg_stat = self.multiple_runs()		
#		
#		return avg_stat
			
###############
## var_names: 'learn_rate','mb_size','seeding', 'k', 'dataname', 'alg_mode', 'eval_mode'	

def check_valid(var_name, var_snapshot, n_iters, n_runs, serial_number = 0, pickle_name = None):
	""" Check if the non-varying parameters in pickle file
		matches that in var_snapshot	
	"""	
	if pickle_name == None:
		pickle_name = 'EXP_STAT_' + var_name + '_s' + str(serial_number) + '.pickle'
	
	if not os.path.exists(pickle_name):
		return True
		
	with open(pickle_name, 'rb') as f:
		curr_stat, meta = pickle.load(f)
	
	# Check meta information
	assert meta['n_iters'] == n_iters and\
		  meta['n_runs'] == n_runs, 'Meta parameters mismatch!'
		  
	# Check the static parameter values	
	for param_name in curr_stat.keys():
		if param_name != var_name:
			if curr_stat[param_name] != var_snapshot[param_name]:
				return False			
			
	return True			
				  

def check_duplicate(var_name, var_value, serial_number = 0, pickle_name = None):
	""" Check if the value is recorded for var_name """
	if pickle_name == None:
		pickle_name = 'EXP_STAT_' + var_name + '_s' + str(serial_number) + '.pickle'
	
	if not os.path.exists(pickle_name):
		print('File does not exist')
		return False
		
	print('Loading %s ...' % pickle_name)
	
	with open(pickle_name, 'rb') as f:
		curr_stat, _ = pickle.load(f)
		
	# check if var_value exists under var_name cell
	if not curr_stat[var_name].has_key(var_value):
		return False
	
	return True	
	

def record_stat(avg_stat, std_stat, var_snapshot, var_name, n_iters, n_runs, \
      serial_number = 0, pickle_name = None, overwrite = False, runtime = None):
	""" 
		Add experiment statistics to EXP_STAT one at a time
	"""	
	
	if pickle_name == None:
		pickle_name = 'EXP_STAT_' + var_name + '_s' + str(serial_number) + '.pickle'
		
	file_exists = True
	if os.path.exists(pickle_name):
		# load it
		with open(pickle_name, 'rb') as f:
			curr_stat, meta = pickle.load(f)	

		# Check if all non-varying parameters in curr_stat matches that in var_snapshot
		assert check_valid(var_name, var_snapshot, n_iters, n_runs, pickle_name),\
		 'The current variable snapshot does not match the pickle file in use!'
		
	else:
		# create an empty dictionary (dictionary of dictionaries)
		print('Creating a new file: %s' % pickle_name)
		meta = dict()
		meta['n_iters'] = n_iters
		meta['n_runs'] = n_runs
		
		
		curr_stat = dict()
		for key in var_snapshot.keys():
			if key == var_name:
				curr_stat[key] = dict()
			else:
				curr_stat[key] = None
				
		file_exists = False
		
	# find the cell for each var_value in var_name
	if file_exists and not overwrite and check_duplicate(var_name, var_snapshot[var_name]):
		
		print('The current variable configuration exists')
		print('Skipping recording..')	
	
	else:
		for param_name in var_snapshot.keys():	
			param_value = var_snapshot[param_name]
			
			# Need this extra line to handle self defined initial centers
			if param_name == 'seed_method' and \
			  type(param_value).__name__ == 'ndarray':
				param_value = 'ndarray'  
			
			print('Storing %s = %s' %(param_name, param_value,))
			if param_name == var_name:
				# Store matrix to the target var_name
				curr_stat[param_name][param_value] = (avg_stat, std_stat, runtime)
			else:
				curr_stat[param_name] = param_value
			
			# pickle updated stat	
			with open(pickle_name, 'wb') as f:
				pickle.dump([curr_stat, meta], f, pickle.HIGHEST_PROTOCOL)
		
	""" for var_value in var_snapshot[var_name]:
	
		if not file_exists:
			print('Storing run stat for %s = %s' % (var_name, var_value,))
			# add var_value to it (overwrite if already exists)	
			curr_stat[var_name][var_value] = avg_stat
		elif not overwrite and check_duplicate(var_name, var_snapshot[var_name]):
			print('The current variable configuration exists')
			print('Skipping recording..')
		else:
			print('Storing run stat to %s = %r' % (var_name, var_value))
			# add var_value to it (overwrite if already exists)	
			curr_stat[var_name][var_value] = avg_stat	"""
	
	
def get_param_values(pickle_name):
	""" Can be used to inspect the variable of a pickle file """
		
	assert os.path.exists(pickle_name), pickle_name + 'does not exist!'
	print('Loading %s ...' % pickle_name)
	
	with open(pickle_name, 'rb') as f:
		curr_stat, meta = pickle.load(f)
	
	print('The n_iters and n_runs are %f and %f' %(meta['n_iters'], meta['n_runs']))
	
	# Find var_name
	var_name = None
	for param_name in curr_stat.keys():
		if type(curr_stat[param_name]) is dict:
			var_name = param_name
			print('The variable %s in this file takes values %s' % (var_name, curr_stat[var_name].keys(),))
		else:
			print('Fixed parameter %s takes value %s' % (param_name, curr_stat[param_name],))	
	# Get keys of dictionary mapped to var_name
	return curr_stat[var_name].keys(), meta	
	
			 

def plot_from_stat(var_name, serial_number = 0, pickle_name = None, display_mode = 'default', var_value_set = None):
	"""Plot cost - time graphs from EXP_STAT """	
	if pickle_name == None:
		pickle_name = 'EXP_STAT_' + var_name + '_s' + str(serial_number) + '.pickle'
	
	assert os.path.exists(pickle_name), pickle_name + 'does not exist!'
	
	print('Loading %s for plotting...' % pickle_name)
	with open(pickle_name, 'rb') as f:
		curr_stat, meta = pickle.load(f)
	
	if var_value_set == None:
		# Get all parameter values under var_name
		var_values = curr_stat[var_name].keys()
	else:
		var_values = var_value_set
	
	# Get meta parameter information
	n_iters = meta['n_iters']
	n_runs = meta['n_runs']
	
	# Transform data if not default display
	if display_mode == 'default' or '(cost - opt_cost)*t':
		x = np.arange(n_iters)
	
	# Add each existing value in var_value_set to plot	
	plots = []
	legends = []
	for var_value in var_values:
		if curr_stat[var_name].has_key(var_value):
			print(var_value)
			#print(len(curr_stat[var_name][var_value]))
			curr_y, var_y, _ = curr_stat[var_name][var_value]
						 
			if display_mode == '(cost - opt_cost)*t':
				curr_y = [ (curr_y[t] - min(curr_y))*t for t in range(len(curr_y))] 
			
			if display_mode == 'default':
				p = plt.errorbar(x, curr_y, yerr = var_y, errorevery = 10)	
			elif display_mode == '(cost - opt_cost)*t':
				curr_y = [ (curr_y[t] - min(curr_y)+1) for t in range(len(curr_y))] 
				p = plt.errorbar(x, curr_y)	
			else:
				curr_y = [ (curr_y[t] - min(curr_y)+1) for t in range(len(curr_y))] 
				p = plt.errorbar(x, curr_y) 
			
			#plt.setp(p, linewidth=4) 
			
			if display_mode == '(cost - opt_cost)*t':
				plt.xscale('log', basex = 2)
			if display_mode == 'log-log':
				plt.xscale('log', basex = 2)
				plt.yscale('log', basey = 2)
			#if display_mode == '(cost - opt_cost)*t':
				#plt.xscale('log', basex = 2)
			
			plots.append(p[0])	
			if var_value == (0,1):
				var_value = 'BB-rate'
			legends.append('The variable %s = %s' %(var_name,var_value,))
			
	if display_mode == 'log-log':
		x = np.arange(1,n_iters)
		y = 30*max(curr_y)/(x)
		p = plt.plot(x,y,'k--')
		plt.xscale('log', basex = 2)
		plt.yscale('log', basey = 2)
		plots.append(p[0])
		legends.append('1/x')
	
		
	# Set legend, labels, axes	
	title_str = 'The effect of varying ' + var_name
	plt.title(title_str)
	plt.legend(plots, legends)
	
	# Plot data
	
##################################################
## Tests / Demo
##################################################
if __name__ == '__main__':
	"""
	''' Test 1: experiment and pickling '''
	# Instanciate experiment with mnist data
	n_clusters = 10
	n_iters = 2001
	n_runs = 3
	exp_demo = MBKM_EXP('mnist', n_clusters, n_iters, n_runs, 
	    alg_mode = 'mbkm', seed_method = 'kmpp', 
	    minibatch_size = 5, learn_rate = (0,1))
	# Set various learning rate params
	var_name = 'learn_rate'
	var_values = {(0,1),(1,2),(1,3),(1,4)}
	# Make validations
	assert check_valid(var_name, exp_demo.var_snapshot, n_iters, n_runs), \
	 'You are using the wrong pickle file for your experiment setup!'
	# Run experiment for each configuration
	for var_value in var_values:
		# Set the target variable value to test
		exp_demo.learn_rate = var_value
		exp_demo.make_var_snapshot()
		# Record non-duplicate values
		if not check_duplicate(var_name, var_value):
			# Run experiment with this configuration
			avg_stat, std_stat, avg_runtime = exp_demo.multiple_runs()	
			# Update and record var_snapshot with avg_stat
			if avg_stat != None:
				record_stat(avg_stat, std_stat, exp_demo.var_snapshot, var_name,\
				   n_iters, n_runs, runtime = avg_runtime, overwrite = True)
	
	''' Test 2: handling of new/repeated configuration '''
	## Run this multiple times to check if repeated values
	## are skipped by experiment
	n_clusters = 10
	n_iters = 2001
	n_runs = 3
	exp_demo = MBKM_EXP('mnist', n_clusters, n_iters, n_runs, 
	    alg_mode = 'mbkm', seed_method = 'kmpp', 
	    minibatch_size = 5, learn_rate = (0,1))
	# Set various learning rate params
	var_name = 'learn_rate'
	var_values = {(0,1),(1,5),(2,4),(2,5),(3,5), (4,5), (6,5)}
	# Make validations
	assert check_valid(var_name, exp_demo.var_snapshot, n_iters, n_runs), \
	 'You are using the wrong pickle file for your experiment setup!'
	
	# Run experiment for each configuration
	for var_value in var_values:
		# Set the target variable value to test
		exp_demo.learn_rate = var_value
		exp_demo.make_var_snapshot()
		
		# Make validations
		if not check_duplicate(var_name, var_value): 
			# Run experiment with this configuration
			avg_stat, std_stat, avg_runtime = exp_demo.multiple_runs()	
			# Update and record var_snapshot with avg_stat
			if avg_stat != None:
				record_stat(avg_stat, std_stat, exp_demo.var_snapshot, var_name ,\
				  n_iters, n_runs, runtime = avg_runtime, overwrite = True)
	      
	''' Test 3: loading from EXP_STAT_learn_rate.pickle and plotting'''
	get_param_values('EXP_STAT_learn_rate_s0.pickle')
	#print var_values, meta['n_iters'], meta['n_runs']
	# Plot all stored values of learning rate
	#plot_from_stat('learn_rate', display_mode = '(cost - opt_cost)*t',\
	#  var_value_set = ((0,1),(1,5),(2,5),(3,5),(4,5), (6,5)))
	#plot_from_stat('learn_rate', display_mode = 'log-log',var_value_set = ((0,1),(1,5),(2,5),\
	#  (3,5),(4,5), (6,5)))
	plot_from_stat('learn_rate', display_mode = 'log-log',var_value_set = ((0,1),(1,5),(2,5),\
	  (3,5),(4,5), (6,5)))
	plt.show()
	# Plot a subset of the learning rate params
	
	# Plot kmcost vs iteration across learning rate params 
	# from stored stat
	"""
	''' Test script'''
	p_name = "EXP_2_lr_fixedcenters_mn_nc10_mb1000"
	plot_from_stat('learn_rate', display_mode = 'log-log', pickle_name = p_name)
	plt.show()
#	from six.moves import cPickle as pickle
#	pickle_name = 'EXP_STAT_learn_rate.pickle'
#	with open(pickle_name, 'rb') as f:
#		curr_stat, meta = pickle.load(f)
	