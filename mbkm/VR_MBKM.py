## Variance-reduced MBKM
## by Cheng Tang, July-2016

import numpy as np
#import matplotlib.pyplot as plt
import random
## from sklearn package
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import k_means_
#from sklearn.metrics.pairwise import pairwise_distances_argmin
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.datasets import fetch_rcv1
#from sklearn import metrics
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import row_norms, squared_norm

## from local modules
from utilities import VR_MB_step
from KMeans_ext import _kmeans_step
from seedingAlgs import setSeed
##########################################################

class VR_MBKM(MiniBatchKMeans):
	def __init__(self, n_clusters=8, init=None, max_iter=100,
	     batch_size=100, verbose=0, compute_labels=True,
	     update_freq=30, learn_rate_params = (0.0,1,0),
	     random_state=None, tol=0.0, max_no_improvement=10,
	     init_size=None, n_init=3, reassignment_ratio=0.01):
	     
		super(VR_MBKM, self).__init__(n_clusters=n_clusters,
		     init=init, max_iter=max_iter, 
		     batch_size=batch_size, verbose=verbose,
		     compute_labels=compute_labels,
		     random_state=random_state, tol=tol,
		     max_no_improvement=max_no_improvement,
		     init_size=init_size, n_init=n_init,
		     reassignment_ratio=reassignment_ratio)
		## set current iteration to 0
		self.curr_iter = 0	## total number of iterations
		## current inner loop iterations
		self.curr_inner_iter = 0 ## equals 0 if not in inner loop
		self.curr_outer_iter = 0
		
		self.counts_ = None
		## we allow changing learning rate definition
		self.learn_rate_params = learn_rate_params
		## mini-batch size
		self.mbsize = batch_size
		## update frequency parameter
		self.update_freq = update_freq
		## Automatically determine whether to precompute distances
		self.precompute_distances = 'auto'
		self.inner_loop = None
		
	def get_curr_iter(self):
		return self.curr_iter
	def set_curr_iter(self, iteration):
		assert type(iteration) is int, "Iteration must be an integer!"
		self.curr_iter = iteration
	
	def set_eta(self):
		c, n, mode = self.learn_rate_params
		if mode == 'constant':
			return float(c)
		else:
			# learning rate is not constant
			# or default
			return float(c) / (self.get_curr_iter() + n)
    
	def set_mbsize(self, m):
		self.mbsize = m
    
	def set_learn_rate_param(self, params):
		self.learn_rate_params = params
	
	def partial_fit(self, D):
		"""
    	Apply one iteration of VR_MBKM
    	
    	Input: self, dataset 
    	Output: self
    	
    	Updated:
    	   -self.curr_iter
    	   -self.curr_inner_iter
    	   -self.tot_inner_iter
    	   -self.cluster_centers_
		"""		
		## perform checks on dataset
		D = check_array(D, accept_sparse ='csr')
    	
		if hasattr(self.init, '__array__'):
			self.init = np.ascontiguousarray(self.init, dtype=np.float64)
    	
		if self.curr_inner_iter == 0:
			self.inner_loop == 0
    	
		if self.curr_iter == 0 or self.inner_loop == 0 or self.update_freq == 0:
    		## OUTER LOOP
    		# use the entire dataset
			X = D
			x_squared_norms = row_norms(X, squared=True)
			self.random_state_ = getattr(self, "random_state_", check_random_state(self.random_state))
                                     
			if self.curr_iter == 0:
    			## initialize centers
				if hasattr(self.init, '__array__'):
					self.cluster_centers_ = self.init
				else:
					self.cluster_centers_ = k_means_._init_centroids(
                        X, self.n_clusters, self.init,
                        random_state=self.random_state_,
                        x_squared_norms=x_squared_norms, init_size=self.init_size)
                    
                    
				_,cost = k_means_._labels_inertia(X, x_squared_norms, self.cluster_centers_)
				#print "Cost of current initial centers on the mini-batch is %r " % cost
            		
				## initialize counts 
				self.counts_ = np.zeros(self.n_clusters, dtype=np.int32)
            
			## this ensures the benchmark centers are either the seeds
			## or obtained from the last iterate of inner loop
			self.benchmark_centers = self.cluster_centers_.copy()	
            	
			## run Lloyd's update with entire data
			distances = np.zeros(X.shape[0], dtype=np.float64)
			self.benchmark_updates, _ , self.squared_diff = _kmeans_step(
                     X=X,x_squared_norms=x_squared_norms,centers=self.benchmark_centers.copy(),
                     distances=distances,precompute_distances=self.precompute_distances,n_clusters=self.n_clusters)
                
			self.cluster_centers_ = self.benchmark_updates.copy()
			self.curr_outer_iter += 1
			self.inner_loop = 1
               
		else:
			## INNER LOOP:
			# use a mini-batch of data
			sample_idx = random.sample(range(D.shape[0]), self.mbsize)
			X = D[sample_idx,:]
			#x_squared_norms = row_norms(X, squared=True)
			self.set_eta() 
			## run VRMB_step with entire data
			distances = np.zeros(X.shape[0], dtype=np.float64)

			self.cluster_centers_, self.squared_diff,_ = VR_MB_step(X, None, 
    		    	self.cluster_centers_.copy(), self.benchmark_centers.copy(),
    		        self.benchmark_updates.copy(),
    		        self.counts_, self.curr_iter, 
    		        np.zeros(0, np.double), 0, distances, random_reassign=False,
                    random_state=self.random_state_, reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose, learn_rate = self.set_eta()) 
    		
			# increment inner loop counts
			self.curr_inner_iter = (self.curr_inner_iter + 1) % self.update_freq
    		
    	
		# increment global loop count
		self.curr_iter += 1
