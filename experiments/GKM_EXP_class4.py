## A class for experimenting with general k-means variants
## Extension from MBKM_EXP_class
## author: Cheng Tang, July, 2016.
########################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
import random
from six.moves import cPickle as pickle
## from local modules
from MBKM_EXP_class import MBKM_EXP
from KMeans_ext import KMeans_ext
from MBKM_ext import MiniBatchKMeans_ext
from VR_MBKM import VR_MBKM
from seedingAlgs import setSeed
from utilities import _compute_inertia

class GKM_EXP(MBKM_EXP):
	def __init__(self, dataname, n_clusters, n_iters, n_runs, 
        alg_mode = 'mbkm', seed_method = 'random', 
	    minibatch_size = None, learn_rate = (0,1), verbose = False):
	    
		super(GKM_EXP, self).__init__(dataname, n_clusters, n_iters, n_runs, 
	      alg_mode = alg_mode, seed_method = seed_method, 
	      minibatch_size = minibatch_size, learn_rate = learn_rate, verbose = verbose)
	    # overide curr_stat
		#self.curr_stat = self.curr_stat / 60000
		self.seeds = None
	
	## get a copy of seeds
	def check_seeds(self):
		return self.seeds
	
	## override
	def multiple_runs(self, record_freq = 1, reuse_seeds = False):
		# reset run counter
		self.n_curr_runs = 0
		accum_stat = np.zeros([self.n_runs, self.n_iters/record_freq])
		accum_runtime = np.zeros([self.n_runs])
		for i in range(self.n_runs):
			if reuse_seeds and self.n_curr_runs > 0:
				self.seed_method = self.check_seeds()
			accum_runtime[i] = self.single_run(record_freq = record_freq)
			accum_stat[i,:] = self.curr_stat
			self.n_curr_runs += 1
	
		return np.mean(accum_stat, 0), np.std(accum_stat,0), np.mean(accum_runtime)
	
	## override
	def single_run(self, record_freq = 1):
		# Reset iteration counter
		self.n_curr_iters = 0
		
		# overide curr_stat
		self.curr_stat = np.zeros(self.n_iters / record_freq)
		
		## Seeding
		self.init_centroids = setSeed(self.train, self.n_clusters, self.xsn_train,\
		  seed_method = self.seed_method)
		  
		self.seeds = self.init_centroids.copy()
		
		if self.verbose and type(self.seed_method) is str:
			print('Successfully initialized %d centroids with %s' % (self.n_clusters,self.seed_method))
		else:
			print('Initialized with self-defined centers')
		## Initialize algorithm
		if self.alg_mode == 'mbkm':
			self.alg = MiniBatchKMeans_ext(n_clusters=self.n_clusters, init=self.init_centroids.copy(),
		      n_init=1,verbose= False, reassignment_ratio = 0)
		elif self.alg_mode == 'km':
			self.alg = KMeans_ext(n_clusters = self.n_clusters, init = self.init_centroids.copy(), n_init = 1,
	  		    verbose = False)
	  	elif self.alg_mode == 'svrgkm':
	  		learn_rate = 1 / float(self.train.shape[0]**(0.5))
	  		update_freq_ = self.train.shape[0]
	  		self.alg = VR_MBKM(n_clusters=self.n_clusters,
	  		     init=self.init_centroids.copy(), update_freq = update_freq_,
	  		     learn_rate_params = (learn_rate, 1, 'constant'),
	  		     n_init=1, verbose=False, reassignment_ratio =0)
	  		self.alg.set_mbsize(self.minibatch_size)
	  	else:
	  		print "algorithm mode %s is not supported!" % self.alg_mode
			return -1
	  		
		runtime = 0  
		print 'number of iterations now is %d' %self.n_iters
		for i in range(self.n_iters):
			runtime = runtime + self.step(record_freq = record_freq)
		if self.verbose:
			print('Finished the %d run of %s' %(self.n_curr_runs+1, self.alg_mode))
			print('Killing algorithm instance..')
		
		self.alg = None
		
		return runtime
		
	## override
	def step(self, record_freq = 1):
		if self.alg_mode == 'mbkm':
			t0 = time()
			sample_idx = random.sample(range(self.train.shape[0]), self.minibatch_size)
			self.alg.partial_fit(self.train[sample_idx,:])
			td = time()
			## Reset algorithm variables
			self.centroids = self.alg.cluster_centers_
		
		elif self.alg_mode == 'km':
			# kmeans
			t0 = time()
			self.alg.partial_fit(self.train)
			td = time()
			## Reset algorithm variables
			self.centroids = self.alg.cluster_centers_
			
		elif self.alg_mode == 'svrgkm':
			t0 = time()
			self.alg.partial_fit(self.train)
			td = time()
			## Reset algorithm variables
			self.centroids = self.alg.cluster_centers_
		else:
			print "algorithm mode %s is not supported!" % self.alg_mode
			return -1
		
		## Record to stat
		if self.n_curr_iters % record_freq == 0:
			if self.eval_mode == 'train':
				eval_data = self.train	
				xsn_eval_data = self.xsn_train	
			else:	
				eval_data = self.test	
				xsn_eval_data = self.xsn_test
			
			self.curr_stat[self.n_curr_iters/record_freq] = _compute_inertia(eval_data,xsn_eval_data,self.centroids)[1]
			
			if self.verbose:
				print('The %s cost at iteration %d is %f' %(self.eval_mode, self.n_curr_iters,self.curr_stat[self.n_curr_iters/record_freq]))
		
		self.n_curr_iters += 1
		return td-t0
## test
if __name__ == '__main__':
	n_clusters = 10
	n_iters = 20*6000
	n_runs = 2
	exp_demo = GKM_EXP('mnist', n_clusters, n_iters, n_runs, 
	    alg_mode = 'mbkm', seed_method = 'random', 
	    minibatch_size = 1, learn_rate = (0.0,1),verbose = True)	

	exp_demo.multiple_runs(record_freq = 6000, reuse_seeds=True)
	stat_mbkm_adalr = exp_demo.curr_stat.copy()
	#exp_demo.alg_mode = 'mbkm'
	#exp_demo.n_iters = 20*6000
	learn_rate = 1 / float(6000**(0.5))
	exp_demo.learn_rate = (learn_rate, 1, 'constant')
	exp_demo.seed_method = exp_demo.check_seeds()
	#exp_demo.learn_rate = (0.0,1)
	#exp_demo.curr_stat = np.zeros(exp_demo.n_iters/6000)
	exp_demo.multiple_runs(record_freq=6000, reuse_seeds=True)
	stat_mbkm_cst = exp_demo.curr_stat.copy()
	##
	c = 4
	t0 = 10
	exp_demo.learn_rate = (c,t0)
	#exp_demo.curr_stat = np.zeros(exp_demo.n_iters/6000)
	exp_demo.seed_method = exp_demo.check_seeds()
	exp_demo.multiple_runs(record_freq=6000,reuse_seeds=True)
	stat_mbkm_linlr = exp_demo.curr_stat.copy()
	##
	exp_demo.alg_mode = 'km'
	exp_demo.n_iters = 20
	#exp_demo.curr_stat = np.zeros(exp_demo.n_iters)
	exp_demo.seed_method = exp_demo.check_seeds()
	exp_demo.multiple_runs(record_freq=1, reuse_seeds=True)
	stat_km = exp_demo.curr_stat.copy()
	filename = 'vrkm_exp2.pickle'
	try:
		with open(filename, 'wb') as f:
				pickle.dump([stat_km, stat_mbkm_cst, stat_mbkm_adalr,stat_mbkm_linlr], f, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save to', filename, ':', e)
	
#    stat_mbkmt = [math.log(i-min(stat_mbkm_linlr)+1) for i in stat_mbkm_linlr]
#    stat_mbkm_adalrt = [math.log(i-min(stat_mbkm_adalr)+1) for i in stat_mbkm_adalr]
#    stat_kmt = [math.log(i-min(stat_km)+1) for i in stat_km]
#    control = [math.log((stat_mbkm_linlr[0]-min(stat_mbkm_linlr))/float(i)) for i in range(1,len(stat_mbkm_linlr)+1)]
#    plt.plot(np.arange(len(stat_mbkm_linlr)), stat_mbkmt, 'r')
#    plt.plot(np.arange(len(stat_mbkm_adalr)), stat_mbkm_adalrt, 'k')
#    plt.plot(np.arange(len(stat_km)), stat_kmt, 'g')
#    plt.plot(np.arange(len(stat_km)), control, 'k--')
#    plt.show()
	