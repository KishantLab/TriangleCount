import dgl
import networkit as nk
import numpy as np
from scipy.sparse import csr_matrix
import random
import torch
import numpy as np
import pandas as pd
import sys
import csv
import time
import torch as th
from scipy.io import mmread
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





csr_example = csr_matrix((np.ones(Edges), (row_ptr, col_idx)), shape=(Nodes, num_nodes))

