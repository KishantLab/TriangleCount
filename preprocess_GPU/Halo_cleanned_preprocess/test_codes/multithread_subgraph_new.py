import dgl
import torch
import concurrent.futures
import time
# Create a random graph for demonstration purposes
start_create = time.time()
num_nodes = 39454746
num_edges = 1581073454
src = torch.randint(0, num_nodes, (num_edges,))
dst = torch.randint(0, num_nodes, (num_edges,))
graph = dgl.graph((src, dst))
print(graph)
G = graph.to('cuda')
end_create = time.time()
# List of node sets for which a subgraph needs to be extracted
node_ids = torch.randint(0, num_nodes, (22324881,))

combined_nodes = torch.tensor(node_ids, device='cuda')

print("graph creation time: ", round((end_create - start_create),4))
# Define a function to extract a subgraph for a given set of nodes
def extract_subgraph(node_ids):
    subgraph = graph.subgraph(node_ids)
    return subgraph

# Use ThreadPoolExecutor for multithreading
# start_subg = time.time()
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     future = executor.submit(extract_subgraph, node_ids)
#     subgraph = future.result()
# end_subg = time.time()

time.sleep(10)

# subgraph now contains the extracted subgraph
# print(subgraph)
# print("Subgraph multithreading creation time: ", round((end_subg - start_subg),4))

start_sg = time.time()
subgraph1 = G.subgraph(combined_nodes)
end_sg = time.time()
print(subgraph1)
print("Subgraph without multithreading creation time: ", round((end_sg - start_sg),4))


