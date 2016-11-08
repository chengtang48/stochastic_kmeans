import numpy as np
import scipy.sparse as sp
import random
import math
from kmpp import kmpp
from helpers.buckshot_robust import buckshot

def setSeed(data, n_clusters, x_squared_norms_data, \
     seed_method = None, m_size = None, n_init = None, verbose = True):
	"""
	Input
		data - ndarray, sparse, or list 
	"""
	init_list = {'random','kmpp','buckshot'}
	
	if n_init == None:
		n_init = 1
	if verbose and type(seed_method) is str:
		print('Initializing %d centers with %s seeding,\
	 	  repeated %d times' % (n_clusters,seed_method, n_init))
		print ('Type of data is %s' % type(data))
		
	
	#if type(data).__name__ != 'ndarray':
		
	#	as_float_array(data)	
	
	if type(seed_method).__name__ != 'ndarray':
		assert seed_method in init_list, 'Unrecognized algorithm!'
	else:
		print seed_method.shape
		n_rows, n_cols = seed_method.shape
		assert n_rows == n_clusters and n_cols == data.shape[1],\
		   'Invalid initial centers are passsed!'
		return seed_method
	
	if seed_method == 'random':
		
		init_centers_indices = range(data.shape[0])
		
		random.shuffle(init_centers_indices)
		init_centers_indices = init_centers_indices[:n_clusters]
		if sp.issparse(data):
			init_centers = data[init_centers_indices].toarray()
		else:
			init_centers = data[init_centers_indices]
		return init_centers
	
	elif seed_method == 'kmpp': # kmpp handles sparse data internally
		return kmpp(X=data, n_clusters=n_clusters,
		  x_squared_norms=x_squared_norms_data, n_local_trials=n_init)
		
	
	else: # buckshot
		if m_size == None:
			n_data = int(math.ceil(n_clusters * math.log(n_clusters)))
		else:
			n_data = m_size
		
		data_indices = range(data.shape[0])
		
		random.shuffle(data_indices)
		data_indices = data_indices[:n_data]
		
		data = data[data_indices,:]
		return buckshot(data, n_clusters, verbose = False)
		
		
## Test
if __name__ == '__main__':
	toy_np = np.arange(12)
	toy_np = np.reshape(toy_np, (6,2))
	xqn_data = np.linalg.norm(toy_np, axis = 1)
	toy = [[0,1],[2,3],[4,5]]
	n_clusters = 2
	n_init = 3
	init_centers = setSeed(toy_np,n_clusters,xqn_data,seed_method = 'buckshot')
	print hasattr(init_centers, '__array__')	