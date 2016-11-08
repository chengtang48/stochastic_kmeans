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
def _mini_batch_update_csr(X, np.ndarray[DOUBLE, ndim=1] x_squared_norms,
                           np.ndarray[DOUBLE, ndim=2] centers,
                           np.ndarray[INT, ndim=1] counts,
                           np.ndarray[INT, ndim=1] nearest_center,
                           np.ndarray[DOUBLE, ndim=1] old_center,
                           int compute_squared_diff,
                           int curr_iter,
                           #np.ndarray[DOUBLE, ndim=1] rand_vec
                           DOUBLE learn_rate
                           ):
    """Incremental update of the centers for sparse MiniBatchKMeans.
    Parameters
    ----------
    X: CSR matrix, dtype float64
        The complete (pre allocated) training set as a CSR matrix.
    centers: array, shape (n_clusters, n_features)
        The cluster centers
    counts: array, shape (n_clusters,)
         The vector in which we keep track of the numbers of elements in a
         cluster
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
        unsigned int n_samples = X.shape[0]
        unsigned int n_clusters = centers.shape[0]
        unsigned int n_features = centers.shape[1]
        DOUBLE eta = learn_rate
        unsigned int iter = curr_iter 
        unsigned int sample_idx, center_idx, feature_idx
        unsigned int k
        unsigned int old_count, new_count
        DOUBLE center_diff
        DOUBLE squared_diff = 0.0
        np.ndarray[DOUBLE,ndim=1] new_center
    

    # move centers to the mean of both old and newly assigned samples
    for center_idx in range(n_clusters):
        old_count = counts[center_idx]
        new_count = old_count

        # count the number of samples assigned to this center
        for sample_idx in range(n_samples):
            if nearest_center[sample_idx] == center_idx:
                new_count += 1

        if new_count == old_count:
            # no new sample: leave this center as it stands
            continue

        # rescale the old center to reflect it previous accumulated weight
        # with regards to the new data that will be incrementally contributed
        if compute_squared_diff:
            old_center[:] = centers[center_idx]
        #centers[center_idx] *= old_count #deleted this line from original code
        ### Added
        new_center = np.zeros(n_features) 
        ###

        # iterate of over samples assigned to this cluster to move the center
        # location by inplace summation
        for sample_idx in range(n_samples):
            if nearest_center[sample_idx] != center_idx:
                continue

            # inplace sum with new samples that are members of this cluster
            # and update of the incremental squared difference update of the
            # center position
            for k in range(X_indptr[sample_idx], X_indptr[sample_idx + 1]):
            ###
            #   centers[center_idx, X_indices[k]] += X_data[k]
                new_center[X_indices[k]]+=X_data[k]    
            ###   
        # inplace rescale center with updated count
        if new_count > old_count:
            # update the count statistics for this center
            counts[center_idx] = new_count
            
            # re-scale the updated center with the total new counts
            #centers[center_idx] /= new_count
            ###
            if eta == 0.0:
                eta = (new_count-old_count)/float(new_count)
                #eta = float(n_samples)/(iter*(new_count-old_count))
                #eta = 1/float(iter*iter)

            centers[center_idx] = centers[center_idx] + eta*(new_center/(new_count-old_count) - centers[center_idx])
            ###
            # update the incremental computation of the squared total
            # centers position change
            if compute_squared_diff:
                for feature_idx in range(n_features):
                    squared_diff += (old_center[feature_idx]
                                     - centers[center_idx, feature_idx]) ** 2

    return centers