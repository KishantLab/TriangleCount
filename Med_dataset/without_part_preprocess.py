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

#---------Convert undirected --------> Directed without partition !!!!-----------#

Find_size = cp.RawKernel(r'''
extern "C" __global__
void Find_size(unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr, unsigned long long int row_ptr_s, unsigned long long int col_idx_s)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
  //if(id==0){bar.set_niter(row_ptr_s+col_idx_s);}
  //__syncthreads();
  if(id < row_ptr_s)
  {
    unsigned long long int count = 0;
    unsigned long long int deg_src = d_row_ptr[id+1] - d_row_ptr[id];
    //unsigned long long int deg_src = d_in_deg[d_org_id[id]];
    //printf(" Deg_src : %llu ",deg_src);
    for(unsigned long long int j=d_row_ptr[id]; j<d_row_ptr[id+1]; j++)
    {
      unsigned long long int deg_dst = d_row_ptr[d_col_idx[j]+1] - d_row_ptr[d_col_idx[j]];
      //unsigned long long int deg_dst = d_in_deg[d_org_id[d_col_idx[j]]];
      //printf("  Deg_dst : %llu ",deg_dst);
      if(deg_src < deg_dst)
      {
        count++;
        //col_idx_Dir[pos] = d_col_idx[j];
        //pos++;
      }
      else if(deg_src == deg_dst)
      {
        if(d_col_idx[j] < id)
        {
          count++;
          //col_idx_Dir[pos] = d_col_idx[j];
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
void Convert(  unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr_sum, unsigned long long int *d_row_ptr_Dir, unsigned long long int *d_col_idx_Dir, unsigned long long int row_ptr_s, unsigned long long int col_idx_s)
{
      unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
      unsigned long long int pos;
      if(id < row_ptr_s)
      {
        if(id==0)
        {
          //bar.set_niter(row_ptr_s+col_idx_s);
          pos = 0;
          d_row_ptr_Dir[0] = 0;
        }
        __syncthreads();
        if(id !=0)
        {
          pos = temp_arr_sum[id-1];
        }
        //unsigned long long int count = 0;
        unsigned long long int deg_src = d_row_ptr[id+1] - d_row_ptr[id];
        //unsigned long long int deg_src = d_in_deg[d_org_id[id]];
        //printf(" Deg_src : %llu ",deg_src);
        for(unsigned long long int j=d_row_ptr[id]; j<d_row_ptr[id+1]; j++)
        {
          unsigned long long int deg_dst = d_row_ptr[d_col_idx[j]+1] - d_row_ptr[d_col_idx[j]];
          //unsigned long long int deg_dst = d_in_deg[d_org_id[d_col_idx[j]]];
          //printf("  Deg_dst : %llu ",deg_dst);
          if(deg_src < deg_dst)
          {
            //count++;
            d_col_idx_Dir[pos] = d_col_idx[j];
            pos++;
          }
          else if(deg_src == deg_dst)
          {
            if(d_col_idx[j] < id)
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

totalTime =0
start = time.time()

file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_extension)
suffix_csr = "_output.csr"
suffix_part = "_part.csr."
file_name = file_name.split("/")
file_name = file_name[len(file_name)-1]
out_filename1 = str(file_name) + suffix_csr
#out_filename2 = str(file_name) + suffix_part + str(sys.argv[2])
print(out_filename1)
#out_filename2 = file_name + suffix_part + str(sys.argv[2])
if file_extension == '.mtx':
    print("Converting mtx2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.graph((u, v))
    #G.add_edges(u, v)
elif file_extension == '.tsv':
    columns = ['Source','Dest','Data']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,low_memory=False)
    print("Converting tsv2dgl..")
    print("This might a take while..")
    u=file['Dest']
    u=np.array(u)
    print(file['Dest'])
    v=file['Source']
    v=np.array(v)
    G = dgl.graph((v,u))
elif file_extension == '.txt':
    columns = ['Source','Dest']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,skiprows=4)
    print("Converting txt2dgl..")
    print("This might a take while..")
    u=file['Dest']
    u=np.array(u)
    print(file['Dest'])
    v=file['Source']
    v=np.array(v)
    G = dgl.graph((v,u))
elif file_extension == '.mmio':
    print("Converting mmio2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.graph((u, v))
    #G.add_edges(u, v)
elif file_name == '.out':
    print("Converting mmio2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.graph((u, v))
    #G.add_edges(u, v)
else:
    print(f"Unsupported file type: {file_extension}")
    exit("If file is TAB Saprated data then remove all comments in file and save it with extention .tsv \n NOTE: only .tsv (Graph Challange), .txt(snap.stanford), .mtx(suit_tensors), .mmio(all) files are supported")

end = time.time()
del u
del v
Nodes = G.num_nodes()
Edges = G.num_edges()

totalTime = totalTime + (end-start)
print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")
#----------------------DGL PREPROCESS-----------------------------------#
size = sys.getsizeof(G)
print(f"The variable 'graph_object' is consuming {size} bytes of memory.")
start = time.time()
print("DGL GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.to_simple(G)
G = dgl.remove_self_loop(G)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory { (mem_usage)} GB,\t {psutil.Process().memory_percent()} % used")

print("DGL SIMPLE GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.add_reverse_edges(G)
size = sys.getsizeof(G)
print(f"The variable 'graph_object' is consuming {size} bytes of memory.")
G = dgl.to_bidirected(G)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB")
print("DGL GRAPH CONSTRUCTION DONE \n",G)
size = sys.getsizeof(G)
print(f"The variable 'graph_object' is consuming {size} bytes of memory.")
isolated_nodes = ((G.in_degrees() == 0) & (G.out_degrees() == 0)).nonzero().squeeze(1)
G.remove_nodes(isolated_nodes)
print(G)
size = sys.getsizeof(G)
print(f"The variable 'graph_object' is consuming {size} bytes of memory.")
Nodes = G.num_nodes()
Edges = G.num_edges()
end = time.time()
totalTime = totalTime + (end-start)
print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
#-------------------------------------------Graph Construction is done ----------#
# start = time.time()
# G = dgl.reorder_graph(G, node_permute_algo='metis',edge_permute_algo='dst', permute_config={'k':5000})
# end = time.time()
# totalTime = totalTime + (end-start)
# print("Reorder is Done !!!!!\t Time of Reorder is :",round((end-start),4), "Seconds")
#-------------------------------------Graph CONSTRUCTION USING MtX----------------#
file = open(out_filename1,'w')
row_ptr_s = len(np.array(G.adj_tensors('csr')[0]))
col_idx_s = len(np.array(G.adj_tensors('csr')[1]))
row_ptr = np.array(G.adj_tensors('csr')[0])
col_idx = np.array(G.adj_tensors('csr')[1])
size = sys.getsizeof(G)
print(f"The variable 'graph_object' is consuming {size} bytes of memory.")
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB \t {psutil.Process().memory_percent()} % used")


print("Converting UNDIR ---> DIR")
size = sys.getsizeof(row_ptr)
print(f"The variable 'row_ptr' is consuming {size} bytes of memory.")
size = sys.getsizeof(col_idx)
print(f"The variable 'col_idx' is consuming {size} bytes of memory.")

start1 = time.time()
d_row_ptr = cp.asarray(row_ptr)
d_col_idx = cp.asarray(col_idx)
temp_arr = cp.empty_like(row_ptr)
N =0
# Call the add kernel function on the GPU
block_size = 1024
grid_size = (row_ptr_s + block_size - 1) // block_size
print(row_ptr_s)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB\t {psutil.Process().memory_percent()} % used")

Find_size((grid_size,), (block_size,), (d_row_ptr, d_col_idx, temp_arr, row_ptr_s, col_idx_s))
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

mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB\t {psutil.Process().memory_percent()} % used")

#Convert<<<nblocks,BLOCKSIZE>>>(d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr_sum, d_row_ptr_Dir, d_col_idx_Dir, in_deg_s, org_id_s, row_ptr_s, col_idx_s);
Convert((grid_size,), (block_size,), (d_row_ptr, d_col_idx, temp_arr_sum, d_row_ptr_Dir, d_col_idx_Dir, row_ptr_s, col_idx_s))

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
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB\t {psutil.Process().memory_percent()} % used")

end1 = time.time()
totalTime = totalTime + (end1-start1)
print("Converting is done !!!!! Time taken: ",round((end1-start1),4))

#--------------------writting CSR to file---------------------------------------#
start = time.time()
file.write("%i " % Nodes)
file.write("%i " % Edges)
file.write("%i " % len(row_ptr_dir))
file.write("%i " % len(col_idx_dir))
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
print(f"Current memory usage: { (mem_usage)} GB\t {psutil.Process().memory_percent()} % used")

#temp_arr.free()
del d_row_ptr
del d_col_idx
del temp_arr
del temp_arr_sum
del d_row_ptr_Dir
del d_col_idx_Dir
del row_ptr
del col_idx
del row_ptr_dir
del col_idx_dir
#cp.cuda.runtime.free(intptr_t temp_arr)
cp._default_memory_pool.free_all_blocks()
file.close()
print("Data Preprocessing is Successfull Total Time taken: ",round(totalTime,4),"Seconds")
