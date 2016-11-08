# cython: profile=True
# Profiling is enabled by default as the overhead does not seem to be measurable
# on this specific use case.
# Modified from sklearn k-means clustering: _kmeans_step.pyx 

from libc.math cimport sqrt
import numpy as np
import scipy.sparse as sp
cimport numpy as np
cimport cython


from sklearn.utils.extmath import norm
#from sklearn.utils.sparsefuncs_fast cimport add_row_csr
from sklearn.utils.fixes import bincount

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _vr_mini_batch_update_csr(X, np.ndarray[DOUBLE, ndim=1] x_squared_norms,
                    	   np.ndarray[DOUBLE, ndim=2] centers,
                           np.ndarray[DOUBLE, ndim=2] benchmark_updates,
                           np.ndarray[INT, ndim=1] counts,
                           np.ndarray[INT, ndim=1] nearest_center_bench,
                           np.ndarray[INT, ndim=1] nearest_center,
                           np.ndarray[DOUBLE, ndim=1] old_center,
                           int compute_squared_diff,
                           int curr_iter,
                           DOUBLE learn_rate
											):
    """Incremental update of the centers for sparse VR-MiniBatchKMeans.
	Parameters
	----------
    X: CSR matrix, dtype float64
        The complete (pre allocated) training set as a CSR matrix.
    centers: array, shape (n_clusters, n_features)
        The cluster centers
    benchmark_updates: array, shape (n_clusters, n_features)
    counts: array, shape (n_clusters,)
         The vector in which we keep track of the numbers of elements in a
         cluster
    nearest_center_bench: array, shape (minibatch size, n_features)
    	 nearest centers of each sample, calculated using benchmark centers
    nearest_center: array, shape (minibatch size, n_features)
         nearest centers of each sample, calculated using current centers
    Returns
    -------
    inertia: float
        The inertia of the batch prior to centers update, i.e. the sum
        distances to the closest center for each sample. This is the objective
        function being minimized by the k-means algorithm.
    squared_diff: float
        The sum of squared update (squared norm of the centers position
        change). If compute_squared_diff is 0, this computation is skipped and
        0.0 is returned instead.
    Both squared diff and inertia are commonly used to monitor the convergence
    of the algorithm.
    """
    cdef:
        np.ndarray[DOUBLE, ndim=1] X_data = X.data
        np.ndarray[int, ndim=1] X_indices = X.indices
        np.ndarray[int, ndim=1] X_indptr = X.indptr
        np.ndarray[DOUBLE, ndim=1] new_center, new_center_bench
        unsigned int n_samples = X.shape[0]
        unsigned int n_clusters = centers.shape[0]
        unsigned int n_features = centers.shape[1]
        DOUBLE eta = learn_rate
        unsigned int iter = curr_iter 
        unsigned int sample_idx, center_idx, feature_idx
        unsigned int k
        unsigned int old_count, new_count, bench_count
        DOUBLE center_diff
        DOUBLE squared_diff = 0.0
    	   
    # move centers, update counts
    for center_idx in range(n_clusters):
        old_count = counts[center_idx]
        new_count = old_count
        bench_count = 0
        # count the number of samples assigned to this center
        for sample_idx in range(n_samples):
            if nearest_center[sample_idx] == center_idx:
                new_count += 1
            if nearest_center_bench[sample_idx] == center_idx:
                bench_count += 1
    	
        if new_count == old_count:
        	# no new sample: leave this center as is
            continue
        
        if compute_squared_diff:
        	# store current center to old center buffer
            old_center[:] = centers[center_idx]
        
		#centers[center_idx] *= old_count #deleted this line from original code
        new_center = np.zeros(n_features)  ## Added lines
        new_center_bench = np.zeros(n_features)

        # iterate of over samples assigned to this cluster to move the center
        # location by inplace summation
        for sample_idx in range(n_samples):
            if nearest_center[sample_idx] != center_idx:
                continue
            
            for k in range(X_indptr[sample_idx], X_indptr[sample_idx + 1]):
                #   centers[center_idx, X_indices[k]] += X_data[k]
                new_center[X_indices[k]]+=X_data[k]    
        
        # update the benchmark center using sample
        if bench_count > 0:
            for sample_idx in range(n_samples):
                if nearest_center_bench[sample_idx] != center_idx:
                    continue
                for k in range(X_indptr[sample_idx], X_indptr[sample_idx + 1]):
                    new_center_bench[X_indices[k]]+=X_data[k]
        else:
            bench_count = 1
        		          
        ## Update count for current center
        #if new_count > old_count:
        # update the count statistics for this center
        # this check may be unnecessary
        
        counts[center_idx] = new_count
            
        #centers[center_idx] /= new_count # deleted line
            
        ## compute learning rate for current center
        if eta == 0.0:
            eta = (new_count-old_count)/float(new_count)
            #eta = float(n_samples)/(iter*(new_count-old_count))
            #eta = 1/float(iter*iter)
			
        ### Current center update:
        centers[center_idx] = centers[center_idx] + eta*(new_center/(new_count-old_count) - centers[center_idx]
                      -new_center_bench/bench_count  + benchmark_updates[center_idx])

        # update the incremental computation of the squared total center change
        if compute_squared_diff:
            for feature_idx in range(n_features):
                squared_diff += (old_center[feature_idx] - centers[center_idx, feature_idx]) ** 2

    return centers