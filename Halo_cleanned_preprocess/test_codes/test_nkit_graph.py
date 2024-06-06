import networkit as nk
import numpy as np
from scipy.sparse import csr_matrix
import random
import time

# Generate a large random graph
num_threads = 8
nk.setNumberOfThreads(num_threads)

num_nodes = 39454746
num_edges = 1581073454

# Use NetworKit's random graph generator for a more efficient large graph creation
start_g = time.time()
G = nk.generators.ErdosRenyiGenerator(num_nodes, num_edges / (num_nodes * (num_nodes - 1)),  directed=True).generate()
print(G)
end_g = time.time()
print("graph creation time: ", round((end_g - start_g),4))
# Alternatively, if you prefer to generate a CSR matrix:
# row = np.random.randint(0, num_nodes, num_edges)
# col = np.random.randint(0, num_nodes, num_edges)
# data = np.ones(num_edges)
# csr_example = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

# To convert the CSR to a NetworKit graph (if using CSR)
# G = nk.nxadapter.nx2nk(nk.graphtools.matrixToGraph(csr_example))

# Select a random subset of 5687 nodes
random.seed(42)
subset_nodes = random.sample(range(num_nodes), 19727373)

# Create induced subgraph from the selected nodes
start_subg = time.time()
# induced_subgraph = nk.graphtools.subgraphAndNeighborsFromNodes(G, subset_nodes, includeInNeighbors=True)
induced_subgraph = nk.graphtools.subgraphFromNodes(G, subset_nodes)

end_subg = time.time()
print(induced_subgraph)
print("Subgraph creation time: ", round((end_subg - start_subg),4))
# Print some basic information about the graphs
print("Original Graph:")
print(f"Nodes: {G.numberOfNodes()}")
print(f"Edges: {G.numberOfEdges()}")

print("\nInduced Subgraph:")
print(f"Nodes: {induced_subgraph.numberOfNodes()}")
print(f"Edges: {induced_subgraph.numberOfEdges()}")

# Optionally, visualize the original and induced subgraph
# Note: Visualizing such large graphs might be challenging and require more resources
# nk.viztasks.drawGraph(G, pos=None, node_size=20)
# nk.viztasks.drawGraph(induced_subgraph, pos=None, node_size=20)

