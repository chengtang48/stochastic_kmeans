# Modify the class MiniBatchKMeans from sklearn
# author: Cheng Tang
import pdb	
#import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from utilities import MB_step

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import fetch_rcv1
from sklearn import metrics
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import row_norms, squared_norm
from time import time
#from . import _k_means

print(__doc__)



class MiniBatchKMeans_ext(MiniBatchKMeans):
    """
    Added parameter: 
    learn_rate: string 'classic', 'theory', callable, or ndarray
    curr_iter: tracks the numb iters executed (counting seeding)
    """
    def __init__(self, n_clusters=8, init=None, max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 learn_rate_params = (0.0,1,None),
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01):
        
        super(MiniBatchKMeans_ext,self).__init__(n_clusters=n_clusters, 
                 init=init, max_iter=max_iter, batch_size=batch_size, 
                 verbose=verbose, compute_labels=compute_labels,
                 random_state=random_state, tol=tol, 
                 max_no_improvement=max_no_improvement,
                 init_size=init_size, n_init=n_init, 
                 reassignment_ratio=reassignment_ratio)
	    
        ## an internal count for current iterations
        self.curr_iter = 0 
        
        ## default learning rate
        self.learn_rate_params = learn_rate_params   
                 
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
            return float(c) / (self.get_curr_iter()+n)
    
    
    def partial_fit(self, X, y=None):
        """Override partial_fit() in MiniBatchKMeans class 
           (Jan-16: added a return var: squared_diff)
           (April-16: changed set_eta as an internal step)
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Coordinates of the data points to cluster.
        """

        X = check_array(X, accept_sparse="csr")
        n_samples, n_features = X.shape
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=np.float64)

        if n_samples == 0:
            return self

        x_squared_norms = row_norms(X, squared=True)
        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))
        if (not hasattr(self, 'counts_')
                or not hasattr(self, 'cluster_centers_')):
            # this is the first call partial_fit on this object:
            # initialize the cluster centers
            # pdb.set_trace()
            if hasattr(self.init, '__array__'):
            	self.cluster_centers_ = self.init
            
            else:
            	self.cluster_centers_ = k_means_._init_centroids(
                  X, self.n_clusters, self.init,
                  random_state=self.random_state_,
                  x_squared_norms=x_squared_norms, init_size=self.init_size)
            
            _,cost = k_means_._labels_inertia(X, x_squared_norms, self.cluster_centers_)
            print "Cost of current initial centers on the mini-batch is %r " % cost
            
            self.counts_ = np.zeros(self.n_clusters, dtype=np.int32)
            #random_reassign = False
            distances = None
            self.curr_iter = 1
        else:
            # The lower the minimum count is, the more we do random
            # reassignment, however, we don't want to do random
            # reassignment too often, to allow for building up counts
            #random_reassign = self.random_state_.randint(
            #    10 * (1 + self.counts_.min())) == 0
            distances = np.zeros(X.shape[0], dtype=np.float64)
            """ modification HERE  """ 
            #self.set_eta() 
            self.cluster_centers_, self.squared_diff,_ = MB_step(X, x_squared_norms, self.cluster_centers_,
                         self.counts_, self.curr_iter, np.zeros(0, np.double), 0,
                         random_reassign=False, distances=distances,
                         random_state=self.random_state_,
                         reassignment_ratio=self.reassignment_ratio,
                         verbose=self.verbose, learn_rate = self.set_eta())              
            self.curr_iter = self.curr_iter+1
        if self.compute_labels:
            self.labels_, self.inertia_ = k_means_._labels_inertia(
                X, x_squared_norms, self.cluster_centers_)

        return self

    
