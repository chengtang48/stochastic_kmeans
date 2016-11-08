# A robust implementation of buckshot algorithm via NetworkX package
import networkx as nx
import numpy as np
import scipy.sparse as sp
import math
## local module
from Linkages_robust import single_link

def buckshot(X, k, verbose = False):
	""" Realization of the seeding algorithm proposed in the
	    Tang-Monteleoni AISTATS-16 paper
		
		params:
		X: should be a n-by-d numpy array, or csr sparse 
		k: number of clusters desired
		verbose: Used for testing purposes
	"""
	### Construct graph from matrix
	graph = nx.Graph()
	
	if sp.issparse(X):
		# handling sparse matrices
		for i in xrange(X.shape[0]):
			vf = i
			for j in xrange(i+1, X.shape[0]):
				vt = j
				sdiff = X.getrow(i)-X.getrow(j)
				w = math.sqrt(sdiff.multiply(sdiff).sum())
				graph.add_edge(vf, vt, weight=w)
	else:
		for i in xrange(X.shape[0]):
			vf = i 
			for j in xrange(i+1, X.shape[0]):
				vt = j 
				w = np.linalg.norm(X[i]-X[j])
				graph.add_edge(vf, vt, weight = w)
	
	if verbose:
		print('Successfully constructed all vertices from input matrix:')
		print graph.nodes()		
	
	# Call single-linkage
	E_new = single_link(graph, k)
	
	new_graph = nx.Graph()
	# Initialize new graph as completely disconnected
	new_graph.add_nodes_from(graph.nodes())
	
	# Construct the sparse edges in new graph	
	for e in E_new:
		vf, vt, w = e
		new_graph.add_edge(vf, vt, weight = w)
	
	if verbose:
		print('Successfully constructed sparse edges from single-link:')
		print new_graph.edges()
				
	""" Compute centroids from new graph """
	
	## first step: collect keys component-wise (find k conn comp)
	## second step: for each component, compute a centroid
	
	centroid_list = np.zeros([k, X.shape[1]])
	ind = 0
	
	for nodes in nx.connected_components(new_graph):
		nodes = list(nodes)
		
		if verbose:
			print nodes	
		
		# the matrix index is the same as the node name
		if sp.issparse(X):
			count = 0
			for sample_idx in nodes:	
				# find the next nonzero column
				if count == 0:
					centroid = X.getrow(sample_idx)
				else:
					centroid = centroid + X.getrow(sample_idx)
				count += 1
			
			centroid = centroid / count
			centroid = centroid.todense()
			centroid_list[ind,:] = centroid
			
		else:
			centroid_list[ind,:] = np.sum(X[nodes], axis = 0) / len(nodes)
		
	
		ind += 1
	
	return centroid_list



##  Test

if __name__ == '__main__':

	toy = np.zeros([4,2])
	toy[1,1] = 2
	toy[2,1] = 1
	toy[0,0] = 3
	toy[3,0] = 1

	print toy
	
	centroids = buckshot(sp.csr_matrix(toy), 2, verbose = False)
	
	for center in centroids:
		print center
 
	
			