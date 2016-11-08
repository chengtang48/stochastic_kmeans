# Utility functions
################### by Cheng Tang (chengtang48@gmail.com), Jan. 2016
import pdb
import numpy as np
import scipy.sparse as sp
import random

from random import gauss
from sklearn.cluster import _k_means
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import metrics
import _MB_step, _VR_MB_step
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import check_random_state
from sklearn.utils.random import choice
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.fixes import astype

print(__doc__)

def _compute_inertia(X, x_squared_norms, centers,
                     precompute_distances=True, distances=None):
    """Compute k-means cost of centers on a dataset"""
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    # distances will be changed in-place
    if sp.issparse(X):
        inertia = _k_means._assign_labels_csr(
            X, x_squared_norms, centers, labels, distances=distances)
    else:
        if precompute_distances:
            return k_means_._labels_inertia_precompute_dense(X, x_squared_norms,
                                                    centers, distances)
        inertia = _k_means._assign_labels_array(
            X, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia
    
    

def MB_step(X, x_squared_norms, centers, counts, curr_iter,
                     old_center_buffer, compute_squared_diff,
                     distances, random_reassign=False,
                     random_state=None, reassignment_ratio=.01,
                     verbose=False, learn_rate = 0.0):
    """Incremental update of the centers for the Minibatch K-Means algorithm.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The original data array.
    x_squared_norms : array, shape (n_samples,)
        Squared euclidean norm of each data point.
    centers : array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE
    counts : array, shape (k,)
         The vector in which we keep track of the numbers of elements in a
         cluster. This array is MODIFIED IN PLACE
    distances : array, dtype float64, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to its closest center.
        May not be None when random_reassign is True.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    random_reassign : boolean, optional
        If True, centers with very low counts are randomly reassigned
        to observations.
    reassignment_ratio : float, optional
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : bool, optional, default False
        Controls the verbosity.
    compute_squared_diff : bool
        If set to False, the squared diff computation is skipped.
    old_center_buffer : int
        Copy of old centers for monitoring convergence.
    
    learn_rate: learning rate
    
    Returns
    -------
    centers: 
    	Updated centers
    inertia : float
        Sum of distances of samples to their closest cluster center.
    squared_diff : numpy array, shape (n_clusters,)
        Squared distances between previous and updated cluster centers.
    """
    # Perform label assignment to nearest centers
    nearest_center, inertia = k_means_._labels_inertia(X, x_squared_norms, centers,
                                              distances=distances)

    if random_reassign and reassignment_ratio > 0:
        random_state = check_random_state(random_state)
        # Reassign clusters that have very low counts
        to_reassign = counts < reassignment_ratio * counts.max()
        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > .5 * X.shape[0]:
            indices_dont_reassign = np.argsort(counts)[int(.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = choice(X.shape[0], replace=False, size=n_reassigns,
                                 random_state=random_state)
            if verbose:
                print("[MiniBatchKMeans] Reassigning %i cluster centers."
                      % n_reassigns)

            if sp.issparse(X) and not sp.issparse(centers):
                assign_rows_csr(X,
                                astype(new_centers, np.intp),
                                astype(np.where(to_reassign)[0], np.intp),
                                centers)
            else:
                centers[to_reassign] = X[new_centers]
        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        counts[to_reassign] = np.min(counts[~to_reassign])
    
    squared_diff = 0.0
    ## implementation for the sparse CSR representation completely written in
    # cython
    if sp.issparse(X):
        if compute_squared_diff:
            old_center_buffer = centers
        #rand_vec = make_rand_vector(X.shape[1]) 
        #learn_rate = 0.0
        centers = _MB_step._mini_batch_update_csr(
            X, x_squared_norms, centers, counts, nearest_center,
            old_center_buffer, compute_squared_diff, curr_iter, learn_rate)
            
        if compute_squared_diff:
            diff = centers - old_center_buffer    
            squared_diff = row_norms(diff,squared=True).sum()
        
        return centers, squared_diff, inertia

    ## dense variant in mostly numpy (not as memory efficient though)
    k = centers.shape[0]
    for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        old_count = counts[center_idx]
        this_count = center_mask.sum()
        counts[center_idx] += this_count # update counts 

        if this_count > 0:
            new_count = counts[center_idx]
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            # inplace remove previous count scaling
            #centers[center_idx] *= counts[center_idx]

            # inplace sum with new points members of this cluster
            #centers[center_idx] += np.sum(X[center_mask], axis=0)

            # update the count statistics for this center
            #counts[center_idx] += count

            # inplace rescale to compute mean of all points (old and new)
            #centers[center_idx] /= counts[center_idx]
            new_center = np.sum(X[center_mask],axis=0)
            if learn_rate == 0.0:
            	learn_rate = (new_count-old_count)/float(new_count)
            
            
            centers[center_idx] = centers[center_idx] + learn_rate*(new_center/(new_count-old_count) - centers[center_idx])

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return centers, squared_diff, inertia

def VR_MB_step(X, x_squared_norms, centers, 
                     benchmark_centers, benchmark_updates, counts, 
                     curr_iter, old_center_buffer, compute_squared_diff,
                     distances, random_reassign=False,
                     random_state=None, reassignment_ratio=.01,
                     verbose=False, learn_rate = 0.0):
    """Incremental update of the centers for the Minibatch K-Means algorithm.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The original data array.
    x_squared_norms : array, shape (n_samples,)
        Squared euclidean norm of each data point.
    centers : array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE
    benchmark_centers: array, shape (k, n_features)
        The benchmark centers 
    benchmark_updates: array, shape (k, n_features)    
        obtained by taking means wrt entire data using benchmark centers
    counts : array, shape (k,)
         The vector in which we keep track of the numbers of elements in a
         cluster. This array is MODIFIED IN PLACE
    distances : array, dtype float64, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to its closest center.
        May not be None when random_reassign is True.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    random_reassign : boolean, optional
        If True, centers with very low counts are randomly reassigned
        to observations.
    reassignment_ratio : float, optional
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : bool, optional, default False
        Controls the verbosity.
    compute_squared_diff : bool
        If set to False, the squared diff computation is skipped.
    old_center_buffer : int
        Copy of old centers for monitoring convergence.
    
    learn_rate: learning rate
    
    Returns
    -------
    centers: 
    	Updated centers
    inertia : float
        Sum of distances of samples to their closest cluster center.
    squared_diff : numpy array, shape (n_clusters,)
        Squared distances between previous and updated cluster centers.
    """
    # Perform label assignment to nearest centers
    nearest_center, inertia = k_means_._labels_inertia(X, x_squared_norms, centers)
    nearest_center_bench, inertia = k_means_._labels_inertia(X, x_squared_norms,benchmark_centers)
    
    #print np.linalg.norm(nearest_center_bench)
    #print ('The difference ratio is %f' % np.dot(nearest_center,nearest_center_bench)**.5)
    
    if random_reassign and reassignment_ratio > 0:
        random_state = check_random_state(random_state)
        # Reassign clusters that have very low counts
        to_reassign = counts < reassignment_ratio * counts.max()
        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > .5 * X.shape[0]:
            indices_dont_reassign = np.argsort(counts)[int(.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = choice(X.shape[0], replace=False, size=n_reassigns,
                                 random_state=random_state)
            if verbose:
                print("[MiniBatchKMeans] Reassigning %i cluster centers."
                      % n_reassigns)

            if sp.issparse(X) and not sp.issparse(centers):
                assign_rows_csr(X,
                                astype(new_centers, np.intp),
                                astype(np.where(to_reassign)[0], np.intp),
                                centers)
            else:
                centers[to_reassign] = X[new_centers]
        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        counts[to_reassign] = np.min(counts[~to_reassign])
    
    squared_diff = 0.0
    ## implementation for the sparse CSR representation completely written in
    # cython
    if sp.issparse(X):
        if compute_squared_diff:
            old_center_buffer = centers
        #rand_vec = make_rand_vector(X.shape[1]) 
        #learn_rate = 0.0
        centers = _VR_MB_step._vr_mini_batch_update_csr(
            X, x_squared_norms, centers.copy(), 
            benchmark_updates.copy(), counts, nearest_center_bench, nearest_center,
            old_center_buffer, compute_squared_diff, curr_iter, learn_rate)
            
        if compute_squared_diff:
            diff = centers - old_center_buffer    
            squared_diff = row_norms(diff,squared=True).sum()
        
        return centers, squared_diff, inertia

    ## dense variant in mostly numpy (not as memory efficient though)
    for center_idx in range(centers.shape[0]):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        center_mask_bench = nearest_center_bench == center_idx
        old_count = counts[center_idx]
        this_count = center_mask.sum()
        counts[center_idx] += this_count # update counts 

        if this_count > 0:
            new_count = counts[center_idx]
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            # inplace remove previous count scaling
            #centers[center_idx] *= counts[center_idx]

            # inplace sum with new points members of this cluster
            #centers[center_idx] += np.sum(X[center_mask], axis=0)

            # update the count statistics for this center
            #counts[center_idx] += count

            # inplace rescale to compute mean of all points (old and new)
            #centers[center_idx] /= counts[center_idx]
            new_center = np.sum(X[center_mask],axis=0)
            ## compute updated centers using benchmark
            if center_mask_bench.sum() == 0:
            	new_center_benchmark = np.zeros(centers.shape[1])
            	this_count_bench = 1
            else:
            	new_center_benchmark = np.sum(X[center_mask_bench], axis=0)
            	this_count_bench = center_mask_bench.sum()
            	
            if learn_rate == 0.0:
            	learn_rate = (new_count-old_count)/float(new_count)
            
            
            centers[center_idx] = centers[center_idx] +\
                 learn_rate*(new_center/(new_count-old_count) - centers[center_idx]
                 - new_center_benchmark/this_count_bench + benchmark_updates[center_idx])

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return centers, squared_diff, inertia

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    a = [x/mag for x in vec]
    a = np.array(a)
    return a