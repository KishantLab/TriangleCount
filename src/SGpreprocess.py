from numpy import array
import torch
import numpy as np
import sys
import dgl

file = open(sys.argv[1],'r')
Lines = file.readlines()
row = Lines[1].strip()
col = Lines[2].strip()
row = list(map(int,row.split(' ')))
col = list(map(int,col.split(' ')))
row = torch.Tensor(row)
row = row.int()
col = torch.Tensor(col)
col = col.int()
G = dgl.graph(('csr', (row,col,[])))
g = G.long()

nopart = int(sys.argv[3])
dgl.distributed.partition_graph(g, 'part', nopart, num_hops=1, part_method='metis',out_path='MetisPart/',reshuffle=True)

file = open(sys.argv[2],'w')
for i in range(nopart):
	g0, nfeats, efeats, partition_book, graph_name, ntypes, etypes  = dgl.distributed.load_partition('MetisPart/part.json', i)
	org_id = list(np.array(g0.ndata['orig_id']))
	SG = g.subgraph(org_id)

	v_arr_s = len(np.array(g0.ndata['inner_node']))
	row_ptr_s = len(np.array(SG.adj_sparse('csr')[0]))
	col_idx_s = len(np.array(SG.adj_sparse('csr')[1]))
	v_arr = np.array(g0.ndata['inner_node'])
	t_ver = np.sum(v_arr)
	row_ptr = np.array(SG.adj_sparse('csr')[0])
	col_idx = np.array(SG.adj_sparse('csr')[1])

	#print(np.array(SG.nodes()))
	#print(np.array(SG.ndata[dgl.NID]))
	#print(np.array(g0.ndata['inner_node']))
	#print(np.array(SG.ndata['inner_node']))
	#sg = to_local(g0.ndata['orig_id'])
	#print(np.array(sg))
	#print(np.array(SG.adj_sparse('csr')[0]))
	#print(np.array(SG.adj_sparse('csr')[1]))

	print("Partitions %i start writting........" %i)

	file.write("%i " % len(np.array(g0.ndata['inner_node'])))
	file.write("%i " % len(np.array(SG.adj_sparse('csr')[0])))
	file.write("%i " % len(np.array(SG.adj_sparse('csr')[1])))
	file.write("%i " % t_ver)
	file.write("\n")
	for data in range(v_arr_s):
		file.write("%i " % v_arr[data])
	file.write("\n")
	for data in range(row_ptr_s):
		file.write("%i " % row_ptr[data])
	file.write("\n")
	for data in range(col_idx_s):
		file.write("%i " % col_idx[data])
	file.write("\n")

	#con = "\n{}\n{}\n{}".format(v_arr,row_ptr,col_idx)
	#file.write(con)
	#file.write("%i " % np.array(g0.ndata['inner_node']))
	#print(len(np.array(g0.ndata['inner_node'])))
	#print(len(np.array(g0.adj_sparse('csr')[0])))
	#print(len(np.array(g0.adj_sparse('csr')[1])))
	#print(np.array(g0.ndata['inner_node']))
	#print(np.array(g0.ndata[dgl.NID]))
	#print(np.array(g0.ndata['orig_id']))
	#print(np.array(g0.adj_sparse('csr')[0]))
	#print(np.array(g0.adj_sparse('csr')[1]))
file.close()
print("File Write Successfull")
