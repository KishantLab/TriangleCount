import dgl
import torch
import concurrent.futures

# Create a random graph for demonstration purposes
num_nodes = 1000
num_edges = 5000
src = torch.randint(0, num_nodes, (num_edges,))
dst = torch.randint(0, num_nodes, (num_edges,))
graph = dgl.graph((src, dst))
print(graph)
# Define a function to extract a subgraph for a given set of nodes
def extract_subgraph(node_ids):
    subgraph = dgl.node_subgraph(graph, node_ids)
    return subgraph

# List of node sets for which subgraphs need to be extracted
node_sets = [torch.randint(0, num_nodes, (100,)) for _ in range(10)]

# Use ProcessPoolExecutor for multiprocessing
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(extract_subgraph, node_set) for node_set in node_sets]
    subgraphs = [future.result() for future in concurrent.futures.as_completed(futures)]

# subgraphs now contains the extracted subgraphs
print(subgraphs)

