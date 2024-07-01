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
import cupy as cp
import psutil
import gc


totalTime =0


file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_name)
print(file_extension)
suffix_csr = "_output.csr"
suffix_part = "_part.csr."
file_name = file_name.split("/")
file_name = file_name[len(file_name)-1]
# out_filename1 = str(file_name) + suffix_csr
# 2 out_filename2 = str(file_name) + suffix_part + str(sys.argv[3])
clean_outfilename2 ="clean_"+str(file_name) + suffix_part + str(sys.argv[2])
# print(out_filename2)

nopart = int(sys.argv[2])

with open(sys.argv[1], 'r') as file:
    line = file.readline().strip()
    Nodes, Edges = map(int, line.split())
    file2 = open(clean_outfilename2,'w')
    file2.write("%i " % Nodes)
    file2.write("%i\n" % Edges)
    total_clean_time = 0.0
    for i in range(nopart):


        # Read the total number of nodes and edges


        # Read the master node, row_ptr_size, col_idx_size, and halo_nodes
        line = file.readline().strip()
        total_node, row_ptr_size, col_idx_size, master_nodes = map(int, line.split())

        # Read the array of row_ptr
        line = file.readline().strip()
        row_ptr = np.array(list(map(int, line.split())))

        # Read the array of col_idx
        line = file.readline().strip()
        col_idx = np.array(list(map(int, line.split())))


        print("Nodes: ",Nodes)
        print("Edges: ",Edges)
        print("total_node: ",total_node)
        print("row_ptr_size: ",row_ptr_size)
        print("col_idx_size: ", col_idx_size)
        print("master_nodes: ", master_nodes)
        print("row_ptr:", row_ptr)
        print("col_idx: ", col_idx)

        G = dgl.graph(('csr', (row_ptr, col_idx, [])))
        d_G_dir = G.to('cuda')
        print("Reading Partiton %i is done !!!!! \n start cleaning...." %i)
        start = time.time()
        # while True:
        for i in range(5):
            start_degree_time = time.time()
            G_indeg = d_G_dir.in_degrees()
            # time.sleep(10)
            end_degree_time = time.time()
            print("Indegree calculation time: ", round((end_degree_time-start_degree_time), 4), "Seconds")
            total_nodes = (d_G_dir.num_nodes())
            # in_deg = []
            # time.sleep(10)
            print("total nodes",total_nodes)
            start_for_loop_time = time.time()
            # for i in range(t_ver,total_nodes):
            #     if G_indeg[i]==0:
            #         in_deg.append(i)
            d_G_indeg = cp.asarray(G_indeg)
            indices = cp.arange(master_nodes, total_nodes)
            # time.sleep(10)
            in_deg = indices[d_G_indeg[indices] == 0]
            # in_deg = zero_indeg_indices.tolist()
            end_for_loop_time = time.time()
            print("For loop execution time: ", round((end_for_loop_time-start_for_loop_time), 4), "Seconds")
            if len(in_deg)==0:
                break
            start_remove_time = time.time()
            d_G_dir.remove_nodes(in_deg)
            # time.sleep(10)
            end_remove_time = time.time()
            print("Node Removal time: ", round((end_remove_time-start_remove_time), 4), "Seconds")
            del G_indeg
            del indices
            # del zero_indeg_indices
            del in_deg
            del d_G_indeg
            # del d_G_dir
            #cp.cu`da.runtime.free(intptr_t temp_arr)
            cp._default_memory_pool.free_all_blocks()
            gc.collect()
            # cp.cuda.runtime.empty_cache()
            # time.sleep(30)
        print(d_G_dir)
        G_dir = d_G_dir.to('cpu')
        d_G_dir = None
        del d_G_dir
        clean_row_ptr = G_dir.adj_tensors('csr')[0]
        clean_col_ptr = G_dir.adj_tensors('csr')[1]
        end = time.time()
        # end_part_time = time.time()
        # total_part_time = total_part_time + (end_part_time - start_part_time)
        total_clean_time = total_clean_time + (end-start)
        totalTime = totalTime + (end-start)
        print("Duplicate edge deletion is done !!!!! Time taken: ",round((end-start),4),"Seconds")
        # Wrriting cleaned data into file
        file2.write("%i " % G_dir.num_nodes())
        file2.write("%i " % len(clean_row_ptr))
        file2.write("%i " % len(clean_col_ptr))
        file2.write("%i " % master_nodes)
        file2.write("\n")
        for data in range(len(clean_row_ptr)):
            file2.write("%i " % clean_row_ptr[data])
        file2.write("\n")
        for data in range(len(clean_col_ptr)):
            file2.write("%i " % clean_col_ptr[data])
        file2.write("\n")

        print("Writing is done !!!!!")
        # totalTime = totalTime + (end-start)
        mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
        print(f"Current memory usage: { (mem_usage)} GB")

        del clean_col_ptr
        del clean_row_ptr
    print("cleaning done !!!!!")
    print("Time taken: ",round(total_clean_time,4),"Seconds")
