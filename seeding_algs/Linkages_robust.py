## Use NetWorkX to implement a more robust version of single link
import networkx as nx


def single_link(graph, k, verbose = False):
	# sort edges 
	sorted_elist = graph.edges(data=True)
	sorted_elist = sorted(sorted_elist, key=lambda (vf, vt, data): data['weight'])
	
	if verbose:
		print sorted_elist
	
	Vlist = graph.nodes()
	E_new = set()
	sim_graph = nx.Graph()
	sim_graph.add_nodes_from(Vlist)	
	
	n_conn_comp = len(Vlist)
	# start adding edges until k components left: time: O(|V||E|)
	for v1, v2, data in sorted_elist:

		if n_conn_comp == k:
			break
		
		# Add an edge if it decreases the number of conn comp
		sim_graph.add_edge(v1, v2, data)
		count = 0
		
		for comp in nx.connected_components(sim_graph):
			count += 1
			
		if count == n_conn_comp - 1:
			E_new.add((v1, v2, data['weight']))
			n_conn_comp -= 1
		else:
			sim_graph.remove_edge(v1, v2)

	return E_new

## Test
if __name__ == '__main__':
	import numpy as np
	
	toy = np.zeros([4,2])
	toy[1,1] = 2
	toy[2,1] = 1
	toy[0,0] = 3
	toy[3,0] = 1
	
	graph = nx.Graph()

	for i in xrange(toy.shape[0]):
		vf = i 
		for j in xrange(i+1, toy.shape[0]):
			vt = j 
			w = np.linalg.norm(toy[i]-toy[j])
			graph.add_edge(vf, vt, weight = w)
	print graph.edges()
	
	print single_link(graph, 2, verbose=True)