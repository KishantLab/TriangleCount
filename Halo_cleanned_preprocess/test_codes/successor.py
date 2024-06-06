import numpy as np
import dgl
import dgl.function as fn
import cupy as cp

# Define the graph in CSR format
indptr = cp.array([0, 2, 3, 6], dtype=cp.int32)
indices = cp.array([1, 2, 0, 1, 2, 3], dtype=cp.int32)

# Transfer CSR data to CPU for creating DGL graph
indptr_cpu = indptr.get()
indices_cpu = indices.get()

# Create a DGL graph from the CSR format
num_nodes = len(indptr_cpu) - 1
rows = np.arange(num_nodes).repeat(np.diff(indptr_cpu))

graph = dgl.graph((indices_cpu, rows), num_nodes=num_nodes)

# Function to find successors of given nodes
def find_successors(graph, nodes):
    with graph.local_scope():
        graph.ndata['out_degree'] = graph.out_degrees()
        return graph.successors(nodes)

# Nodes for which we want to find successors
nodes = cp.array([0, 1, 2], dtype=cp.int32)

# Find successors using the GPU
successors = find_successors(graph, nodes.get())

# Convert the result to CuPy array
successors_cupy = cp.array(successors)

# Print the result
print("Successors:", successors_cupy)

