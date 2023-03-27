from numpy import array
import torch
import numpy as np
import pandas as pd
import sys
import dgl
import csv
import pymetis
import time
from dgl.distgnn import partition
from dgl.distgnn.partition.libra_partition import partition_graph
from dgl.distgnn.partition.libra_partition import libra_partition
from dgl.distgnn.tools import load_proteins
import torch as th
from scipy.io import mmread
import random
totalTime =0

#from dgl/distgnn/partition/libra_partition import *
#from dgl/distgnn/tools/tools import *
#-------------------------------------Graph CONSTRUCTION USING MtX----------------#
'''
totalTime =0
start = time.time()
print("Start File Reading")
file = open(sys.argv[1],'r')
Lines = file.readlines()
print("File Reading done !!!!! \nstart line spliting...")
row = Lines[1].strip()
col = Lines[2].strip()
row = list(map(int,row.split(' ')))
col = list(map(int,col.split(' ')))
print("spliting done !!!!! \nStart Converting in Tensor...")
row = torch.Tensor(row)
row = row.int()
col = torch.Tensor(col)
col = col.int()
print("Converting Tensor is done !!!!! \nStart construction of graph....")


#start = time.time()
# row = torch.Tensor(row_ptr_dir)
# row = row.int()
# col = torch.Tensor(col_idx_dir)
# col = col.int()
G = dgl.graph(('csr', (row,col,[])))
g = G.long()
print(g)
end = time.time()
totalTime = totalTime + (end-start)
'''
#-------------------------------tootls---------------------------------------------
print("Converting mtx2dgl..")
print("This might a take while..")
a_mtx = mmread(sys.argv[1])
coo = a_mtx.tocoo()
u = th.tensor(coo.row, dtype=th.int64)
v = th.tensor(coo.col, dtype=th.int64)
g = dgl.DGLGraph()

g.add_edges(u, v)
n = g.number_of_nodes()
feat_size = 128  ## arbitrary number
feats = th.empty([n, feat_size], dtype=th.float32)

## arbitrary numbers
train_size = 1000000
test_size = 500000
val_size = 5000
nlabels = 256

train_mask = th.zeros(n, dtype=th.bool)
test_mask = th.zeros(n, dtype=th.bool)
val_mask = th.zeros(n, dtype=th.bool)
label = th.zeros(n, dtype=th.int64)

for i in range(train_size):
    train_mask[i] = True

for i in range(test_size):
    test_mask[train_size + i] = True

for i in range(val_size):
    val_mask[train_size + test_size + i] = True

for i in range(n):
    label[i] = random.choice(range(nlabels))

g.ndata["feat"] = feats
g.ndata["train_mask"] = train_mask
g.ndata["test_mask"] = test_mask
g.ndata["val_mask"] = val_mask
g.ndata["label"] = label

##-------------------------------------------------------tools end-----------------
#end = time.time()
#totalTime = totalTime + (end-start)
#print("Construction Graph using Directed Successfull!!!! \t Time Taken :",round((end-start),4), "Seconds")
print("Start Partitioning.....")
start = time.time()
nopart = int(sys.argv[3])
#partition_graph(nopart, G, 'MetisPart/')
libra_partition(nopart, g, 'MetisPart/')

print("Partititon is done !!!!! \nstart reading partition from file.... ")
file = open(sys.argv[2],'w')
for i in range(nopart):
    g0, nfeats, efeats, partition_book, graph_name, ntypes, etypes  = dgl.distributed.load_partition('MetisPart/part.json', i)
    print("Reading Partiton %i is done !!!!! \n start coverting CSR....")
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

    print("Converting is done !!!!! \n Partitions %i start writting........" %i)

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
