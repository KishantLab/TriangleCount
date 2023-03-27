
from numpy import array
import torch
import numpy as np
import pandas as pd
import sys
import dgl
import csv
import pymetis
import time
import torch as th
from scipy.io import mmread
import random
import os
from tqdm import tqdm

#-------------------------------------Graph CONSTRUCTION USING MtX----------------#
file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_extension)

'''
print("Converting mtx2dgl..")
print("This might a take while..")
a_mtx = mmread(sys.argv[1])
coo = a_mtx.tocoo()
u = th.tensor(coo.row, dtype=th.int64)
v = th.tensor(coo.col, dtype=th.int64)
G = dgl.graph()
G.add_edges(u, v)
'''
totalTime =0
start = time.time()
#file=pd.read_csv(sys.argv[1],delimiter='\t')

columns = ['Source','Dest','Data']
file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,skiprows=1)

print("Data Load From File ..\n")
dest=file['Dest']
dest=np.array(dest)
print(file['Dest'])
source=file['Source']
source=np.array(source)
end = time.time()
totalTime = totalTime + (end-start)
print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")
print("Construct Graph Using DGL.\n")
start = time.time()

G = dgl.graph((source,dest))
print("DGL GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.to_simple(G)
G = dgl.remove_self_loop(G)
print("DGL SIMPLE GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.add_reverse_edges(G)
G = dgl.to_bidirected(G)
print("DGL GRAPH CONSTRUCTION DONE \n",G)

isolated_nodes = ((G.in_degrees() == 0) & (G.out_degrees() == 0)).nonzero().squeeze(1)
G.remove_nodes(isolated_nodes)
print(G)

in_deg = np.array(G.in_degrees())
in_deg_s = len(np.array(G.in_degrees()))
print(in_deg)

end = time.time()
totalTime = totalTime + (end-start)
print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
#-------------------------------------------Graph Construction is done ----------#

#-------------------------------------DGL METIS GRAPH PARTITIONING------------------------#
#nopart = 2
nopart = int(sys.argv[3])
print("Start Partitioning.....")
start = time.time()
#n_cuts, node_parts = pymetis.part_graph(nopart, adjacency=adjacency_list)
#nodes_part = dgl.metis_partition_assignment(G, nopart, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
#parts = dgl.metis_partition(g, k, reshuffle=True, mode='k-way')
#parts = dgl.metis_partition(G, nopart, reshuffle=True)
node_parts = dgl.metis_partition_assignment(G,nopart)
end = time.time()
totalTime = totalTime + (end-start)
print("Partition is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

#nodes_part = np.argwhere(np.array(membership) == i).ravel()
print("Partitions Contructions with halo nodes ..")
start = time.time()
parts, orig_nids, orig_eids=dgl.partition_graph_with_halo(G, node_parts, 1, reshuffle=True)
end = time.time()
totalTime = totalTime + (end-start)
print("Halo Node CONSTRUCTION is Done !!!!!\t Time of construction is :",round((end-start),4), "Seconds")
file = open(sys.argv[2],'w')
file.write("%i " % len(in_deg))
file.write("\n")
for data in tqdm(range(len(in_deg)),desc="Writing Degree",leave=False):
    file.write("%i " % in_deg[data])
file.write("\n")
for i in range(nopart):
    #g0, nfeats, efeats, partition_book, graph_name, ntypes, etypes  = dgl.distributed.load_partition('MetisPart/part.json', i)
    print("Reading Partiton %i is done !!!!!" %i)
    start = time.time()
    #org_id = list(np.array(g0.ndata['orig_id']))
    org_id = list(np.array(parts[i].ndata['orig_id']))
    SG = G.subgraph(org_id)
    #SG = dgl.node_subgraph(G, org_id)
    #n_id = np.array(SG.ndata[dgl.NID])
    v_arr = np.array(parts[i].ndata['inner_node'])
    #len(np.array(parts[i].ndata['inner_node']))
    t_ver = np.sum(v_arr)
    #v_arr_s = len(np.array(parts[i].ndata['inner_node']))
    row_ptr_s = len(np.array(SG.adj_sparse('csr')[0]))
    col_idx_s = len(np.array(SG.adj_sparse('csr')[1]))
    row_ptr = (SG.adj_sparse('csr')[0])
    col_idx = (SG.adj_sparse('csr')[1])
    #print(org_id)
    #print(SG.nodes())
    #print()
    #print(row_ptr)
    #print(col_idx)

    start = time.time()
    file.write("%i " % len(org_id))
    file.write("%i " % len(row_ptr))
    file.write("%i " % len(col_idx))
    file.write("%i " % t_ver)
    file.write("\n")
    for data in tqdm(range(len(org_id)),desc="Writing ORGINAL_ID",leave=False):
        file.write("%i " % org_id[data])
    file.write("\n")
    for data in tqdm(range(len(row_ptr)),desc="Writing ROW_POINTER",leave=False):
        file.write("%i " % row_ptr[data])
    file.write("\n")
    for data in tqdm(range(len(col_idx)),desc="Writing COL_INDEX",leave=False):
        file.write("%i " % col_idx[data])
    file.write("\n")
    end = time.time()
    print("Writing is done !!!!! Time taken: ",round((end-start),4),"Seconds")
    totalTime = totalTime + (end-start)
    #print(np.array(SG.nodes()))
    #print(np.array(SG.ndata[dgl.NID]))
    #print(np.array(g0.ndata['inner_node']))
    #print(np.array(SG.ndata['inner_node']))
    #sg = to_local(g0.ndata['orig_id'])
    #print(np.array(sg))
    #print(np.array(SG.adj_sparse('csr')[0]))
    #print(np.array(SG.adj_sparse('csr')[1]))

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
print("Data Preprocessing is Successfull Total Time taken: ",round(totalTime,4),"Seconds")
