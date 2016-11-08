# Extend the class KMeans to include partial_fit() method
import time
import pdb
import logging
import numpy as np
import scipy.sparse as sp
#import matplotlib.pyplot as plt
from sklearn.cluster import k_means_
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import fetch_rcv1
from sklearn import metrics
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, squared_norm
from time import time
from sklearn.cluster import _k_means

print(__doc__)

def _kmeans_step(X, x_squared_norms, centers,
                    distances,
                    precompute_distances,
                    n_clusters,
                    random_state=None):
    """Incremental update of the centers for the Minibatch K-Means algorithm.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        The original data array.
        x_squared_norms : array, shape (n_samples,)
        Squared euclidean norm of each data point.
        centers : array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE
        distances : array, dtype float64, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to its closest center.
        May not be None when random_reassign is True.
        random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
        
        Returns
        -------
        inertia : float
        Sum of distances of samples to their closest cluster center.
        squared_diff : numpy array, shape (n_clusters,)
        Squared distances between previous and updated cluster centers.
    """
    centers_old = centers.copy()
    # labels assignment is also called the E-step of EM
    labels, inertia = k_means_._labels_inertia(X, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

    # computation of the means is also called the M-step of EM
    if sp.issparse(X):
        centers = _k_means._centers_sparse(X, labels, n_clusters,
                                               distances)
    else:
        
        centers = _k_means._centers_dense(X, labels, n_clusters, distances)
    """       if best_inertia is None or inertia < best_inertia:
              best_labels = labels.copy()
              best_centers = centers.copy()
              best_inertia = inertia
    """
    shift = squared_norm(centers_old - centers)
    """        if shift <= tol:
            if verbose:
                print("Converged at iteration %d" % i)

            break

    if shift > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
    """                                                                                                
    return centers,inertia, shift



class KMeans_ext(KMeans):
   def __init__(self, n_clusters=8, init=None, max_iter=100,
                verbose=0, random_state=None, tol=0.0, n_init=3):
         super(KMeans_ext,self).__init__(
            n_clusters=n_clusters, init = init, max_iter = max_iter,
            verbose = verbose, random_state = random_state, tol=tol, n_init=n_init)
                 
                 
                 
   def partial_fit(self, X):
    #Update k means estimate on a single iteration.

        X = check_array(X, accept_sparse="csr")
        n_samples, n_features = X.shape
        x_squared_norms = row_norms(X, squared=True) #currently has redundancy
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=np.float64)
        
        #       if n_samples == 0:
        #            return self

        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))
        #      if (not hasattr(self, 'counts_')
        #              or not hasattr(self, 'cluster_centers_')):
        if (not hasattr(self, 'cluster_centers_')):
            # this is the first call partial_fit on this object:
            # initialize the cluster centers
            self.cluster_centers_ = k_means_._init_centroids(
                X, self.n_clusters, self.init,
                random_state=self.random_state_,
                x_squared_norms=x_squared_norms)
            print("Initialization complete")
        #           self.counts_ = np.zeros(self.n_clusters, dtype=np.int32)
        #            random_reassign = False
            distances = None
            """        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia(
                X, x_squared_norms, self.cluster_centers_)
            """
            return self
        else:
            """ # The lower the minimum count is, the more we do random
            # reassignment, however, we don't want to do random
            # reassignment too often, to allow for building up counts
            random_reassign = self.random_state_.randint(
                10 * (1 + self.counts_.min())) == 0
            """
            distances = np.zeros(X.shape[0], dtype=np.float64)

            """  _mini_batch_step(X, x_squared_norms, self.cluster_centers_,
                self.counts_, np.zeros(0, np.double), 0
                random_reassign=random_reassign, distances=distances,
                random_state=self.random_state_,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)
            """
    
            self.cluster_centers_,self.inertia_ , squared_diff = _kmeans_step(
                X=X,x_squared_norms=x_squared_norms,centers=self.cluster_centers_,
                distances=distances,precompute_distances=self.precompute_distances,n_clusters=self.n_clusters)

            """        if self.compute_labels:
                self.labels_, self.inertia_ = _labels_inertia(
                X, x_squared_norms, self.cluster_centers_)
            """
            return self, squared_diff

    
