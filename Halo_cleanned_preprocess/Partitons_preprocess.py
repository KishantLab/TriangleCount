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

Find_size = cp.RawKernel(r'''
extern "C" __global__
void Find_size(unsigned long long int *d_in_deg, unsigned long long int *d_org_id, unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr, unsigned long long int in_deg_s, unsigned long long int org_id_s, unsigned long long int row_ptr_s, unsigned long long int col_idx_s)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    if(id < row_ptr_s)
    {
      unsigned long long int count = 0;
      //unsigned long long int deg_src = row_ptr[i+1] - row_ptr[i];
      unsigned long long int deg_src = d_in_deg[d_org_id[id]];
      //printf(" Deg_src : %llu ",deg_src);
      for(unsigned long long int j=d_row_ptr[id]; j<d_row_ptr[id+1]; j++)
      {
        //unsigned long long int deg_dst = row_ptr[col_index[j]+1] - row_ptr[col_index[j]];
        unsigned long long int deg_dst = d_in_deg[d_org_id[d_col_idx[j]]];
        //printf("  Deg_dst : %llu ",deg_dst);
        if(deg_src < deg_dst)
        {
          count++;
          //col_idx_Dir[pos] = col_idx[j];
          //pos++;
        }
        else if(deg_src == deg_dst)
        {
          if(d_org_id[d_col_idx[j]] < d_org_id[id])
          {
            count++;
            //col_idx_Dir[pos] = col_idx[j];
             //pos++;
          }
        }
        //bar.update();
        temp_arr[id] = count;
        //printf("%llu",count);
        //row_ptr_Dir[i+1] = pos;
      }
    }
}
'''
, 'Find_size')

Convert = cp.RawKernel(r'''
extern "C" __global__
void Convert(unsigned long long int *d_in_deg, unsigned long long int *d_org_id, unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr_sum, unsigned long long int *d_row_ptr_Dir, unsigned long long int *d_col_idx_Dir, unsigned long long int in_deg_s, unsigned long long int org_id_s, unsigned long long int row_ptr_s, unsigned long long int col_idx_s)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    unsigned long long int pos;
    if(id < row_ptr_s)
    {
        if(id==0)
        {
        pos = 0;
        d_row_ptr_Dir[0] = 0;
        }
        __syncthreads();
        if(id !=0)
        {
        pos = temp_arr_sum[id-1];
        }
        unsigned long long int deg_src = d_in_deg[d_org_id[id]];
        //printf(" Deg_src : %llu ",deg_src);
        for(unsigned long long int j=d_row_ptr[id]; j<d_row_ptr[id+1]; j++)
        {
        unsigned long long int deg_dst = d_in_deg[d_org_id[d_col_idx[j]]];
        //printf("  Deg_dst : %llu ",deg_dst);
        if(deg_src < deg_dst)
        {
          //count++;
          d_col_idx_Dir[pos] = d_col_idx[j];
          pos++;
        }
        else if(deg_src == deg_dst)
        {
          if(d_org_id[d_col_idx[j]] < d_org_id[id])
          {
            //count++;
            d_col_idx_Dir[pos] = d_col_idx[j];
            pos++;
          }
        }
        //bar.update();
        //temp_arr[id] = count;
        d_row_ptr_Dir[id+1] = pos;
        }
    }
}
''', 'Convert')

#-------------------------------------Graph CONSTRUCTION USING data----------------#
totalTime =0
start = time.time()

file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_name)
print(file_extension)
suffix_csr = "_output.csr"
suffix_part = "_part.csr."
file_name = file_name.split("/")
file_name = file_name[len(file_name)-1]
out_filename1 = str(file_name) + suffix_csr
out_filename2 = str(file_name) + suffix_part + str(sys.argv[3])
clean_outfilename2 ="clean_"+str(file_name) + suffix_part + str(sys.argv[3])
print(out_filename2)

with open(sys.argv[1], 'r') as file:
    # Read the total number of nodes and edges
    line = file.readline().strip()
    Nodes, Edges = map(int, line.split())

    # Read the master node, row_ptr_size, col_idx_size, and halo_nodes
    line = file.readline().strip()
    master_node, row_ptr_size, col_idx_size, halo_nodes = map(int, line.split())

    # Read the array of row_ptr
    line = file.readline().strip()
    row_ptr = np.array(list(map(int, line.split())))

    # Read the array of col_idx
    line = file.readline().strip()
    col_idx = np.array(list(map(int, line.split())))

    # Read the array of in_deg
    line = file.readline().strip()
    in_deg = np.array(list(map(int, line.split())))

# with open(sys.argv[2], 'r') as file:
#     # Read nodes parts
#     line = file.readline().strip()
#     node_parts = np.array(list(map(int, line.split())))

print("Nodes: ",Nodes)
print("Edges: ",Edges)
print("Master Nodes: ",master_node)
print("row_ptr_size: ",row_ptr_size)
print("col_idx_size: ", col_idx_size)
print("halo_nodes: ", halo_nodes)
print("row_ptr:", row_ptr)
print("col_idx: ", col_idx)
# print(node_parts)
# print(type(row_ptr))
# print(type(col_idx))

d_in_deg = cp.asarray(in_deg)
in_deg_s = len(in_deg)

G = dgl.graph(('csr', (row_ptr, col_idx, [])))
print(G)

# Create DGL graph from CSR
# g = dgl.graph((graph_data['col_idx'], np.repeat(np.arange(graph_data['row_ptr_size']-1), np.diff(graph_data['row_ptr']))))
# print(g)
# print(type(G))
total_part_time = 0.0
# print("Partitions Contructions with halo nodes ..")
# start = time.time()
# parts, orig_nids, orig_eids = dgl.partition_graph_with_halo(G, node_parts, 1, resuffle)
# parts, orig_nids, orig_eids=dgl.partition_graph_with_halo(G, node_parts, 1, reshuffle=False)
# end = time.time()
# totalTime = totalTime + (end-start)
# print("Halo Node CONSTRUCTION is Done !!!!!\t Time of construction is :",round((end-start),4), "Seconds")
# file = open('test.csr','w')
# file.write("%i " % Nodes)
# file.write("%i\n" % Edges)
file2 = open(clean_outfilename2,'w')
file2.write("%i " % Nodes)
file2.write("%i\n" % Edges)
# end_part_time = time.time()
# total_part_time = total_part_time + (end_part_time - start_part_time)
total_clean_time = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# G = G.to(device)
# nopart = int(sys.argv[3])
with open(sys.argv[2], 'r') as file:
    nopart = int(sys.argv[3])

    for i in range(nopart):
        print("Reading Partiton %i is done !!!!! \n start coverting CSR...." %i)
        line = file.readline().strip()
        total_master_nodes = int(line.strip())
        # total_master_nodes =  int(total_master_nodes)

        line = file.readline().strip()
        combined_nodes_s = int(line.strip())

        line = file.readline().strip()
        combined_nodes = np.array(list(map(int, line.split())))
        # combined_nodes = torch.tensor(combined_nodes, device=device)

        print("master nodes: ",total_master_nodes)
        print("combined nodes: ",len(combined_nodes))

# for i in range(nopart):
    #g0, nfeats, efeats, partition_book, graph_name, ntypes, etypes  = dgl.distributed.load_partition('MetisPart/part.json', i)

        # g = G.to('cpu')
        start = time.time()
        start_part_time = time.time()
        #org_id = list(np.array(g0.ndata['orig_id']))
        # org_id = list(np.array(parts[i].ndata['orig_id']))
        # print(parts[1].nodes())
        # SG = G.subgraph(org_id)
        # SG = parts[i]
        mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
        print(f"Current memory usage: { (mem_usage)} GB")

        # org_id = np.array(parts[i].ndata[NID])
        # print(org_id)
        # org_id_s = len(org_id)
        #SG = dgl.node_subgraph(G, org_id)
        # org_id = np.array(SG.ndata[dgl.NID])
        start_subgraph_time = time.time()
        SG = G.subgraph(combined_nodes)
        end_subgraph_time = time.time()
        print("Subgraph creation time: ", round((end_subgraph_time-start_subgraph_time),4))
        print(SG)
        # SG = SG.to('cpu')
        # G = G.to('cpu')

        org_id = np.array(SG.ndata[dgl.NID])
        # print("org_id: ",org_id)
        org_id_s = len(org_id)
        # print(n_id)
        inner_node = torch.zeros(combined_nodes.shape[0], dtype=torch.bool)
        inner_node[:total_master_nodes] = True  # Mark induced nodes as True
        # SG.ndata['inner_node'] = inner_node
        del combined_nodes
        # v_arr = np.array(SG.ndata['inner_node'])
        inner_node = np.array(inner_node)
        v_arr = inner_node.astype(int)
        # print("v_arr: ",v_arr)
        #len(np.array(parts[i].ndata['inner_node']))
        t_ver = np.sum(v_arr)
        v_arr_s = len(v_arr)
        row_ptr_s = len(np.array(SG.adj_tensors('csr')[0]))
        col_idx_s = len(np.array(SG.adj_tensors('csr')[1]))
        row_ptr = np.array(SG.adj_tensors('csr')[0])
        col_idx = np.array(SG.adj_tensors('csr')[1])
        # Sort the column indices within each row range specified by row_ptr
        for x in range(len(row_ptr) - 1):
            col_idx[row_ptr[x]:row_ptr[x + 1]] = np.sort(col_idx[row_ptr[x]:row_ptr[x + 1]])
        mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
        print(f"Current memory usage: { (mem_usage)} GB")

        print("Converting UNDIR ---> DIR")
        start1 = time.time()

        d_org_id = cp.asarray(org_id)
        d_row_ptr = cp.asarray(row_ptr)
        d_col_idx = cp.asarray(col_idx)
        temp_arr = cp.empty_like(row_ptr)
        N =0
        # Call the add kernel function on the GPU
        block_size = 1024
        grid_size = (row_ptr_s + block_size - 1) // block_size
        print(row_ptr_s)
        Find_size((grid_size,), (block_size,), (d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr, in_deg_s, org_id_s, row_ptr_s, col_idx_s))
        # Print the result
        #print(temp_arr)
        temp_arr_sum = cp.cumsum(temp_arr)
        #print(temp_arr_sum)
        temp_arr_sum_s = len(temp_arr_sum)

        d_row_ptr_Dir = cp.empty_like(d_row_ptr)
        #d_col_idx_Dir = cp.empty_like(d_col_idx)
        dtype = type(col_idx)
        N = int(temp_arr_sum[temp_arr_sum_s-1])
        d_col_idx_Dir = cp.empty(N, dtype=col_idx.dtype)

        #Convert<<<nblocks,BLOCKSIZE>>>(d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr_sum, d_row_ptr_Dir, d_col_idx_Dir, in_deg_s, org_id_s, row_ptr_s, col_idx_s);
        Convert((grid_size,), (block_size,), (d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr_sum, d_row_ptr_Dir, d_col_idx_Dir, in_deg_s, org_id_s, row_ptr_s, col_idx_s))

        # create host array to hold selected elements
        num_elements = N
        col_idx_dir = np.empty(num_elements, dtype=d_col_idx_Dir.dtype)
        # print(type(col_idx_dir))
        # print(len(col_idx_dir))

        # copy selected elements from device array to host array
        start_index = 0
        end_index = start_index + num_elements
        col_idx_dir[start_index:end_index] = d_col_idx_Dir[:N].get()

        # row_ptr_dir = cp.asnumpy(d_row_ptr_Dir)
        # col_idx_dir = cp.asnumpy(d_col_idx_Dir)
        row_ptr_dir = d_row_ptr_Dir.get()
        #col_idx_dir = d_col_idx_Dir.get()
        print(len(col_idx))
        print(len(col_idx_dir))

        end1 = time.time()
        totalTime = totalTime + (end1-start1)
        print("Converting is done !!!!! Time taken: ",round((end1-start1),4))
        mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
        print(f"Current memory usage: { (mem_usage)} GB")

        del d_org_id
        del d_row_ptr
        del d_col_idx
        del temp_arr
        del temp_arr_sum
        del d_row_ptr_Dir
        del d_col_idx_Dir
        del org_id
        del inner_node
        del row_ptr
        del col_idx
        cp._default_memory_pool.free_all_blocks()
        # cp.cuda.runtime.empty_cache()
        gc.collect()

        time.sleep(30)
        #--------------------writting CSR to file---------------------------------------#
        start = time.time()
        G_dir = dgl.graph(('csr', (row_ptr_dir, col_idx_dir, [])))
        print(G_dir)
        d_G_dir = G_dir.to('cuda')
        del G_dir
        time.sleep(30)
        while True:
            start_degree_time = time.time()
            G_indeg = d_G_dir.in_degrees()
            time.sleep(10)
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
            indices = cp.arange(t_ver, total_nodes)
            time.sleep(10)
            zero_indeg_indices = indices[d_G_indeg[indices] == 0]
            in_deg = zero_indeg_indices.tolist()
            end_for_loop_time = time.time()
            print("For loop execution time: ", round((end_for_loop_time-start_for_loop_time), 4), "Seconds")
            if len(in_deg)==0:
                break
            start_remove_time = time.time()
            d_G_dir.remove_nodes(in_deg)
            time.sleep(10)
            end_remove_time = time.time()
            print("Node Removal time: ", round((end_remove_time-start_remove_time), 4), "Seconds")
            del G_indeg
            del indices
            del zero_indeg_indices
            del in_deg
            del d_G_indeg
            # del d_G_dir
            #cp.cu`da.runtime.free(intptr_t temp_arr)
            cp._default_memory_pool.free_all_blocks()
            gc.collect()
            # cp.cuda.runtime.empty_cache()
            # time.sleep(30)
        print(d_G_dir)
        # G_dir = d_G_dir.to('cpu')
        d_G_dir = None
        del d_G_dir
        clean_row_ptr = G_dir.adj_tensors('csr')[0]
        clean_col_ptr = G_dir.adj_tensors('csr')[1]
        end = time.time()
        end_part_time = time.time()
        total_part_time = total_part_time + (end_part_time - start_part_time)
        total_clean_time = total_clean_time + (end-start)
        totalTime = totalTime + (end-start)
        print("Duplicate edge deletion is done !!!!! Time taken: ",round((end-start),4),"Seconds")
        # Wrriting cleaned data into file
        file2.write("%i " % G_dir.num_nodes())
        file2.write("%i " % len(clean_row_ptr))
        file2.write("%i " % len(clean_col_ptr))
        file2.write("%i " % t_ver)
        file2.write("\n")
        for data in range(len(clean_row_ptr)):
            file2.write("%i " % clean_row_ptr[data])
        file2.write("\n")
        for data in range(len(clean_col_ptr)):
            file2.write("%i " % clean_col_ptr[data])
        file2.write("\n")

        #-----Wrriting Original data into file------
        # start = time.time()
        # file.write("%i " % total_master_nodes)
        # file.write("%i " % len(row_ptr_dir))
        # file.write("%i " % len(col_idx_dir))
        # file.write("%i " % t_ver)
        # file.write("\n")
        # #for data in range(v_arr_s):
        # #file.write("%i " % v_arr[data])
        # #file.write("\n")
        # for data in range(len(row_ptr_dir)):
        #     file.write("%i " % row_ptr_dir[data])
        # file.write("\n")
        # for data in range(len(col_idx_dir)):
        #     file.write("%i " % col_idx_dir[data])
        # file.write("\n")
        # end = time.time()
        print("Writing is done !!!!! Time taken: ",round((end-start),4),"Seconds")
        totalTime = totalTime + (end-start)
        mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
        print(f"Current memory usage: { (mem_usage)} GB")
        
        del clean_col_ptr
        del clean_row_ptr
        #temp_arr.free()
        del G_dir 
        del row_ptr_dir
        del col_idx_dir
        #cp.cuda.runtime.free(intptr_t temp_arr)
        # cp.cuda.runtime.empty_cache()
        cp._default_memory_pool.free_all_blocks()
del d_in_deg
file.close()
print("Total Clean Time: ", round(total_clean_time,4), "Seconds")
print("Total Partition Time: ", round(total_part_time,4), "Seconds")
print("Data Preprocessing is Successfull Total Time taken: ",round(totalTime,4),"Seconds")
