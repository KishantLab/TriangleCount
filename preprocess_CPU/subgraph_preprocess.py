from numpy import array
import torch
import numpy as np
import pandas as pd
import sys
import dgl
import csv
import time
import torch as th
from scipy.io import mmread
import random
import os
from tqdm import tqdm
import psutil

#-------------------------------------Graph CONSTRUCTION USING MtX----------------#

totalTime =0
start = time.time()

file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_extension)
suffix_csr = "_output.csr"
suffix_part = "_part.csr."
file_name = file_name.split("/")
file_name = file_name[len(file_name)-1]
out_filename1 = str(file_name) + suffix_csr
out_filename2 = str(file_name) + suffix_part + str(sys.argv[2])
print(out_filename2)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} bytes")

if file_extension == '.mtx':
    print("Converting mtx2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.DGLGraph()
    G.add_edges(u, v)
elif file_extension == '.tsv':
    columns = ['Source','Dest','Data']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns)
    print("Converting tsv2dgl..")
    print("This might a take while..")
    dest=file['Dest']
    dest=np.array(dest)
    print(file['Dest'])
    source=file['Source']
    source=np.array(source)
    G = dgl.graph((source,dest))
elif file_extension == '.txt':
    columns = ['Source','Dest']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,skiprows=4)
    print("Converting txt2dgl..")
    print("This might a take while..")
    dest=file['Dest']
    dest=np.array(dest)
    print(file['Dest'])
    source=file['Source']
    source=np.array(source)
    G = dgl.graph((source,dest))
elif file_extension == '.mmio':
    print("Converting mmio2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.DGLGraph()
    G.add_edges(u, v)
elif file_extension == '.tsv_1':
    columns = ['Source','Dest']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,low_memory=False,skiprows=1)
    print("Converting tsv2dgl..")
    print("This might a take while..")
    u=file['Dest']
    u=np.array(u)
    print(file['Dest'])
    v=file['Source']
    v=np.array(v)
    G = dgl.graph((v,u))
else:
    print(f"Unsupported file type: {file_extension}")
    exit("If file is TAB Saprated data then remove all comments in file and save it with extention .tsv \n NOTE: only .tsv (Graph Challange), .txt(snap.stanford), .mtx(suit_sparse), .mmio(all) files are supported")
end = time.time()
totalTime = totalTime + (end-start)
print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")

start = time.time()
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
in_deg_s = len(in_deg)
print(in_deg)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB")
Nodes = G.num_nodes()
Edges = G.num_edges()
end = time.time()
end = time.time()
totalTime = totalTime + (end-start)
print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
#-------------------------------------------Graph Construction is done ----------#

#-------------------------------------DGL METIS GRAPH PARTITIONING------------------------#
#nopart = 2
nopart = int(sys.argv[2])
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
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} bytes")
#nodes_part = np.argwhere(np.array(membership) == i).ravel()
print("Partitions Contructions with halo nodes ..")
start = time.time()
parts, orig_nids, orig_eids=dgl.partition_graph_with_halo(G, node_parts, 1, reshuffle=True)
end = time.time()
totalTime = totalTime + (end-start)
print("Halo Node CONSTRUCTION is Done !!!!!\t Time of construction is :",round((end-start),4), "Seconds")
file = open(out_filename2,'w')
file.write("%i " % Nodes)
file.write("%i\n" % Edges)
for i in range(nopart):
    #g0, nfeats, efeats, partition_book, graph_name, ntypes, etypes  = dgl.distributed.load_partition('MetisPart/part.json', i)
    print("Reading Partiton %i is done !!!!! \n start coverting CSR...." %i)
    start = time.time()
    #org_id = list(np.array(g0.ndata['orig_id']))
    org_id = list(np.array(parts[i].ndata['orig_id']))
    SG = G.subgraph(org_id)
    #SG = dgl.node_subgraph(G, org_id)
    #n_id = np.array(SG.ndata[dgl.NID])
    v_arr = np.array(parts[i].ndata['inner_node'])
    #len(np.array(parts[i].ndata['inner_node']))
    t_ver = np.sum(v_arr)
    v_arr_s = len(np.array(parts[i].ndata['inner_node']))
    row_ptr_s = len(np.array(SG.adj_sparse('csr')[0]))
    col_idx_s = len(np.array(SG.adj_sparse('csr')[1]))
    row_ptr = (SG.adj_sparse('csr')[0])
    col_idx = (SG.adj_sparse('csr')[1])
    #print(org_id)
    #print(SG.nodes())
    #print()
    #print(row_ptr)
    #print(col_idx)
    print("Find Degree of all nodes")
    start1 = time.time()

    end1 = time.time()
    totalTime = totalTime + (end1-start1)
    print("Converting is done !!!!! Time taken: ",round((end1-start1),4))
    col_idx_dir =[]
    row_ptr_dir =[]
    count =0
    row_ptr_dir.append(count)
    #--------------------------------CONVERT UNDIRECTED SUBGRAPH TO DIRECTED--------------#
    for k in tqdm(range(len(org_id)), desc="Converting Directed"):
        #deg_src = row_ptr[k+1] - row_ptr[k]
        #deg_src = G.in_degree(org_id[k])
        deg_src = in_deg[org_id[k]]
        j = row_ptr[k]
        #pbar = tqdm(total=row_ptr[k+1],leave=False)
        while j<row_ptr[k+1]:
            #deg_dst = row_ptr[col_idx[j]+1] - row_ptr[col_idx[j]]
            #deg_dst = G.in_degree(org_id[col_idx[j]])
            deg_dst = in_deg[org_id[col_idx[j]]]
            if (deg_src < deg_dst):
                col_idx_dir.append(col_idx[j])
                count = count+1
                #pbar.update(1)
            elif (deg_src == deg_dst):
                if(org_id[col_idx[j]] < org_id[k]):
                    col_idx_dir.append(col_idx[j])
                    count = count+1
                    #pbar.update(1)
            j = j+1
        row_ptr_dir.append(count)
        #pbar.close()

    col_idx_dir = np.array(col_idx_dir)
    row_ptr_dir = np.array(row_ptr_dir)
    #print(row_ptr_dir)
    #print(col_idx_dir)
    print(t_ver)
    print(len(row_ptr))
    print(len(col_idx))
    print(len(row_ptr_dir))
    print(len(col_idx_dir))
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} GB")

    #--------------------writting CSR to file---------------------------------------#

    #row_ptr = np.array(SG.adj_sparse('csr')[0])
    #col_idx = np.array(SG.adj_sparse('csr')[1])
    end = time.time()
    totalTime = totalTime + (end-start)
    print("Converting is done !!!!! Time taken: ",round((end-start),4)," Seconds \n Partitions %i start writting........" %i)
    start = time.time()
    file.write("%i " % v_arr_s)
    file.write("%i " % len(row_ptr_dir))
    file.write("%i " % len(col_idx_dir))
    file.write("%i " % t_ver)
    file.write("\n")
    #for data in range(v_arr_s):
        #file.write("%i " % v_arr[data])
    #file.write("\n")
    for data in range(len(row_ptr_dir)):
        file.write("%i " % row_ptr_dir[data])
    file.write("\n")
    for data in range(len(col_idx_dir)):
        file.write("%i " % col_idx_dir[data])
    file.write("\n")
    end = time.time()
    print("Writing is done !!!!! Time taken: ",round((end-start),4),"Seconds")
    totalTime = totalTime + (end-start)
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} GB")
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
