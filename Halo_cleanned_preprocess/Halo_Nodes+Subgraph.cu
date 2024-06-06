#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <cusparse_v2.h>
#include <sys/time.h>
#include <cub/cub.cuh> 
using namespace std;

#define BLOCK_SIZE 1024

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		printf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

struct CustomOpT {
            __host__ __device__ bool operator()(unsigned long long a, unsigned long long b) const {
                // Define your custom comparison logic here
                return a < b; // Example: ascending order
            }
        };


__device__ __forceinline__ unsigned long long int Search (unsigned long long int skey , unsigned long long int *neb, unsigned long long int sizelist)
{
	unsigned long long int total = 0;
	if(skey < neb[0] || skey > neb[sizelist])
	{
		return 0;
	}
	else if(skey == neb[0] || skey == neb[sizelist])
	{
		return 1;
	}
	else
	{
		unsigned long long int lo=1;
		unsigned long long int hi=sizelist-1;
		unsigned long long int mid=0;
		while( lo <= hi)
		{
			mid = (hi+lo)/2;
			//printf("\nskey :%d , mid : %d ",skey,neb[mid]);
			if( neb[mid] < skey){lo=mid+1;}
			else if(neb[mid] > skey){hi=mid-1;}
			else if(neb[mid] == skey)
			{
				total++;
				break;
			}
		}
	}
	return total;
}

__global__ void find_halo_nodes1(unsigned long long *d_row_ptr_G, unsigned long long *d_col_idx_G,
                                signed long long *d_halo_nodes, unsigned long long induced_nodes_s,
                                unsigned long long *d_induced_nodes, unsigned long long *d_pos, unsigned long long num_edges)
{
    unsigned long long bid = blockIdx.x;
    if (bid < induced_nodes_s)
    {
        unsigned long long node = d_induced_nodes[bid];
        unsigned long long start = d_row_ptr_G[node];
        unsigned long long end = d_row_ptr_G[node + 1];
        unsigned long long deg = end - start;
        for (unsigned long long i = threadIdx.x; i < deg; i+= BLOCK_SIZE)
        {
            // unsigned long long flag = 0;
            // unsigned long long pos = 0;
            unsigned long long flag = Search(d_col_idx_G[start+i], d_induced_nodes, induced_nodes_s-1);
            // for (unsigned long long j = 0; j < induced_nodes_s; j++)
            // {
            //     if (d_col_idx_G[start+i] == d_induced_nodes[j])
            //     {
            //         flag = 1;
            //         break;
            //     }
            // }
            // printf("%llu ", flag);
            if (flag == 0)
            {
                // pos = d_pos[0];
                // bool flag2 = 1;
                // for (unsigned long long k = 0; k<=d_pos[0]; k++){
                //     if(d_col_idx_G[start+i]==d_halo_nodes[k]){
                //         flag2 = 0;
                //         break;
                //     }
                // }
                // __syncthreads();
                // if(flag2==1){
                    unsigned long long pos = atomicAdd(d_pos, 1);
                    // if(i==0 || i==deg-1){
                    //     printf("Bid: %llu, Tid: %llu, d_pos: %llu \n", bid, i, d_pos[0]);
                    // }
                    if(pos<num_edges && d_halo_nodes[pos]==-1){
                        d_halo_nodes[pos] = d_col_idx_G[start+i];
                    }
            }
                // printf("%llu ", d_halo_nodes[pos]);
            }
        }
}

__global__ void Find_size(unsigned long long int *d_in_deg, unsigned long long int *d_h_combined_nodes, unsigned long long int *d_row_ptr, 
                            unsigned long long int *d_col_idx, unsigned long long int *temp_arr, unsigned long long int in_deg_s, 
                            unsigned long long int h_combined_nodes_s, unsigned long long int row_ptr_s, unsigned long long int col_idx_s,
                            unsigned long long *d_filtered_halo_nodes_out, unsigned long long filtered_halo_nodes_size,
                            unsigned long long *d_induced_nodes, unsigned long long indices_size)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    if(id < h_combined_nodes_s)
    {
      unsigned long long int count = 0;
      //unsigned long long int deg_src = row_ptr[i+1] - row_ptr[i];
      unsigned long long src = d_h_combined_nodes[id];
      unsigned long long int deg_src = d_in_deg[src];
      //printf(" Deg_src : %llu ",deg_src);
      for(unsigned long long int j=d_row_ptr[src]; j<d_row_ptr[src+1]; j++)
      {
        //unsigned long long int deg_dst = row_ptr[col_index[j]+1] - row_ptr[col_index[j]];
        unsigned long long dest = d_col_idx[j];
        unsigned long long int deg_dst = d_in_deg[dest];
        //printf("  Deg_dst : %llu ",deg_dst);
        if(deg_src < deg_dst)
        {
          unsigned long long flag1 = Search(dest, d_filtered_halo_nodes_out, filtered_halo_nodes_size-1);
          unsigned long long flag2 = Search(dest, d_induced_nodes, indices_size-1);
          if(flag1==1 || flag2==1){
            count++;
          }
          //col_idx_Dir[pos] = col_idx[j];
          //pos++;
        }
        else if(deg_src == deg_dst)
        {
          if(dest<src)
          {
          unsigned long long flag1 = Search(dest, d_filtered_halo_nodes_out, filtered_halo_nodes_size-1);
          unsigned long long flag2 = Search(dest, d_induced_nodes, indices_size-1);                    
          if(flag1==1 || flag2==1){
                        count++;
                    }            
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

__global__
void Convert(unsigned long long int *d_in_deg, unsigned long long int *d_h_combined_nodes, unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr_sum, unsigned long long int *d_row_ptr_Dir, unsigned long long int *d_col_idx_Dir, 
            unsigned long long int in_deg_s, unsigned long long int org_id_s, unsigned long long int h_combined_nodes_s, unsigned long long int col_idx_s,
            unsigned long long *d_index_array, unsigned long long *indegree, unsigned long long *d_filtered_halo_nodes_out, unsigned long long filtered_halo_nodes_size,
            unsigned long long *d_induced_nodes, unsigned long long indices_size)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    unsigned long long int pos;
    if(id < h_combined_nodes_s)
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
        unsigned long long src = d_h_combined_nodes[id];
        unsigned long long int deg_src = d_in_deg[src];
        //printf(" Deg_src : %llu ",deg_src);
        for(unsigned long long int j=d_row_ptr[src]; j<d_row_ptr[src+1]; j++)
        {
        unsigned long long int dest = d_col_idx[j];
        unsigned long long int deg_dst = d_in_deg[dest];
        //printf("  Deg_dst : %llu ",deg_dst);
        if(deg_src < deg_dst)
        {
          //count++;
          unsigned long long flag1 = Search(dest, d_filtered_halo_nodes_out, filtered_halo_nodes_size-1);
          unsigned long long flag2 = Search(dest, d_induced_nodes, indices_size-1);            
        //   printf("Src: %llu, Dest: %llu, Index: %llu \n", src, dest, d_index_array[dest]);
            if(flag1==1 || flag2==1){
                    d_col_idx_Dir[pos] = d_index_array[dest];
                    pos++;
                    atomicAdd(&indegree[dest], 1);
            }
        }
        else if(deg_src == deg_dst)
        {
          if(dest < src)
          {
            //count++;
          unsigned long long flag1 = Search(dest, d_filtered_halo_nodes_out, filtered_halo_nodes_size-1);
          unsigned long long flag2 = Search(dest, d_induced_nodes, indices_size-1);            
        //   printf("Src: %llu, Dest: %llu, Index: %llu \n", src, dest, d_index_array[dest]);
            if(flag1==1 || flag2==1){
                    d_col_idx_Dir[pos] = d_index_array[dest];
                    pos++;
                    atomicAdd(&indegree[dest], 1);
            }
          }
        }
        d_row_ptr_Dir[id+1] = pos;
        }
    }
}

void create_subgraph_with_halo(unsigned long long nopart, const unsigned long long *node_parts,
                               const unsigned long long *row_ptr_G,
                               const unsigned long long *col_idx_G,
                               const unsigned long long *in_deg,
                               const string &output_filename,
                               unsigned long long num_nodes,
                               unsigned long long num_edges)
{
    ofstream outfile(output_filename);
    if (!outfile.is_open())
    {
        cerr << "Error opening file: " << output_filename << endl;
        return;
    }
    outfile <<num_nodes <<" "<<num_edges<<endl;
    double total_kernel_time = 0.0;

    for (unsigned long long value = 0; value < nopart; ++value)
    {
        cout << value << "th Partitions Constructions with halo nodes .." << endl;

        cout << "Finding Indices..." << endl;

        // Finding indices of the partition
        unsigned long long *indices = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
        unsigned long long indices_size = 0;
        for (long long int i = 0; i < num_nodes; ++i) {
            if (node_parts[i] == value) {
                indices[indices_size++] = i;
            }
        }

        unsigned long long induced_nodes_s = indices_size;
        unsigned long long *d_induced_nodes;
        cudaMalloc(&d_induced_nodes, induced_nodes_s * sizeof(unsigned long long));
        cudaMemcpy(d_induced_nodes, indices, induced_nodes_s * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        unsigned long long *d_pos;
        cudaMalloc(&d_pos, sizeof(unsigned long long));
        cudaMemset(d_pos, 0, sizeof(unsigned long long));

        signed long long *d_halo_nodes;
        cudaMalloc(&d_halo_nodes, num_edges * sizeof(signed long long));
        cudaMemset(d_halo_nodes, -1, num_edges * sizeof(signed long long));


        // cout << "Indices: ";
        // for (unsigned long long i = 0; i < indices_size; ++i) {
        //     cout << indices[i] << " ";
        // }
        // cout << endl;

        unsigned long long *d_row_ptr_G;
        unsigned long long *d_col_idx_G;
        cudaMalloc(&d_row_ptr_G, (num_nodes+1) * sizeof(unsigned long long));
        cudaMalloc(&d_col_idx_G, num_edges * sizeof(unsigned long long));
        cudaMemcpy(d_row_ptr_G, row_ptr_G, (num_nodes+1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx_G, col_idx_G, num_edges * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        // unsigned long long grid_size_G = ((num_nodes+1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cout << "Number of induced nodes: " << induced_nodes_s << endl;
        struct timeval begin2, end2;
        gettimeofday(&begin2, 0);
        // find_halo_nodes<<<grid_size_G, BLOCK_SIZE>>>(d_row_ptr_G, d_col_idx_G, d_halo_nodes, induced_nodes_s, d_induced_nodes, d_pos);
        find_halo_nodes1<<<induced_nodes_s, BLOCK_SIZE>>>(d_row_ptr_G, d_col_idx_G, d_halo_nodes, induced_nodes_s, d_induced_nodes, d_pos, num_edges);
        cudaDeviceSynchronize();
        cudaFree(d_pos);
        gettimeofday(&end2, 0);
        long seconds = end2.tv_sec - begin2.tv_sec;
        long microseconds = end2.tv_usec - begin2.tv_usec;
        double elapsed2 = seconds + microseconds*1e-6;            
        cout<<"Time measured for Halo Nodes creation: "<<elapsed2<<" seconds."<<endl;
        cout << "Kernel code completed" << endl;
        total_kernel_time += elapsed2;
        // Retrieve halo nodes from device
        signed long long *h_halo_nodes = (signed long long *)malloc(num_edges * sizeof(signed long long));
        cudaMemcpy(h_halo_nodes, d_halo_nodes, num_edges * sizeof(signed long long), cudaMemcpyDeviceToHost);        
        unsigned long long *filtered_halo_nodes = (unsigned long long *)malloc(num_edges * sizeof(unsigned long long));
        unsigned long long filtered_halo_nodes_size = 0;
        for (unsigned long long i = 0; i < num_edges; ++i) {
            if (h_halo_nodes[i] != -1) {
                filtered_halo_nodes[filtered_halo_nodes_size++] = h_halo_nodes[i];
            }
            else break;
        }
        cudaFree(d_halo_nodes);
        // cout<<"Filtered nodes Collected"<<endl;
        // cout << "Halo nodes (raw): ";
        // for (unsigned long long i = 0; i<num_nodes; i++) {
        //     if (h_halo_nodes[i] != -1) {
        //         cout << h_halo_nodes[i] << " ";
        //     }
        //     else break;
        // }
        // cout << endl;

        // Remove duplicates
        void *d_temp_storage1 = NULL;
        size_t temp_storage_bytes = 0;

        unsigned long long *d_filtered_halo_nodes;
        cudaMalloc(&d_filtered_halo_nodes, filtered_halo_nodes_size * sizeof(unsigned long long));
        cudaMemcpy(d_filtered_halo_nodes, filtered_halo_nodes, filtered_halo_nodes_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        CustomOpT custom_op;
        struct timeval begin3, end3;
        gettimeofday(&begin3, 0);
        cub::DeviceMergeSort::SortKeys(d_temp_storage1, temp_storage_bytes, d_filtered_halo_nodes, filtered_halo_nodes_size, custom_op);
    
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage1, temp_storage_bytes);
    
        // Run sorting operation
        cub::DeviceMergeSort::SortKeys(d_temp_storage1, temp_storage_bytes, d_filtered_halo_nodes, filtered_halo_nodes_size, custom_op);

        unsigned long long *d_filtered_halo_nodes_out;
        cudaMalloc(&d_filtered_halo_nodes_out, filtered_halo_nodes_size * sizeof(unsigned long long)); 
        // cudaMemset(d_filtered_halo_nodes_out, 0, filtered_halo_nodes_size * sizeof(unsigned long long));

        unsigned long long *d_filtered_halo_nodes_out_s;
        cudaMalloc(&d_filtered_halo_nodes_out_s, sizeof(unsigned long long));
        // cudaMemset(d_filtered_halo_nodes_out_s, 0, sizeof(unsigned long long));

        void *d_temp_storage2 = NULL;
         
    
        cub::DeviceSelect::Unique(d_temp_storage2, temp_storage_bytes, d_filtered_halo_nodes, d_filtered_halo_nodes_out, d_filtered_halo_nodes_out_s, filtered_halo_nodes_size);
    
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage2, temp_storage_bytes);
    
        // Run selection
        cub::DeviceSelect::Unique(d_temp_storage2, temp_storage_bytes, d_filtered_halo_nodes, d_filtered_halo_nodes_out, d_filtered_halo_nodes_out_s, filtered_halo_nodes_size);

        unsigned long long *unique_halo_nodes_s = (unsigned long long *)malloc(sizeof(unsigned long long));
        cudaMemcpy(unique_halo_nodes_s, d_filtered_halo_nodes_out_s, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(d_filtered_halo_nodes);

        cout<<unique_halo_nodes_s[0]<<endl;
        printf("Unique value find done\n");

        unsigned long long *unique_halo_nodes = (unsigned long long *)malloc(filtered_halo_nodes_size*sizeof(unsigned long long));
        cudaMemcpy(unique_halo_nodes, d_filtered_halo_nodes_out, filtered_halo_nodes_size*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        printf("cuda memcpy done\n");
        cudaFree(d_temp_storage1);
        cudaFree(d_temp_storage2);

        // for(unsigned long long i=0; i<unique_halo_nodes_s[0]; i++)
        // {
        //     printf("%llu ",unique_halo_nodes[i]);
        // }
        // printf("\n");
        // printf(" halo node :%llu\n",) /
        // sort(filtered_halo_nodes, filtered_halo_nodes + filtered_halo_nodes_size);
        // auto end_unique = unique(filtered_halo_nodes, filtered_halo_nodes + filtered_halo_nodes_size);
        // filtered_halo_nodes_size = distance(filtered_halo_nodes, end_unique);
        gettimeofday(&end3, 0);
        long seconds2 = end3.tv_sec - begin3.tv_sec;
        long microseconds2 = end3.tv_usec - begin3.tv_usec;
        double elapsed3 = seconds2 + microseconds2*1e-6;            
        cout<<"Time measured for duplication removal: "<<elapsed3<<" seconds."<<endl;
        total_kernel_time += elapsed3;

        // Combine induced subgraph and halo nodes
        unsigned long long h_combined_nodes_s = indices_size + unique_halo_nodes_s[0];
        unsigned long long *h_combined_nodes = (unsigned long long *)malloc(h_combined_nodes_s * sizeof(unsigned long long));
        copy(indices, indices + indices_size, h_combined_nodes);
        copy(unique_halo_nodes, unique_halo_nodes + unique_halo_nodes_s[0], h_combined_nodes + indices_size);
        // cout << "Number of Combined nodes: "<<h_combined_nodes_s<<endl;
        // for (unsigned long long i = 0; i < h_combined_nodes_s; ++i) {
        //     cout << h_combined_nodes[i] << " ";
        // }
        // cout << endl;

        // Prepare result buffers

        // unsigned long long *d_indegree;
        // cudaMalloc((void **)&d_indegree, h_combined_nodes_s * sizeof(unsigned long long));
        // cudaMemset(d_indegree, 0, h_combined_nodes_s * sizeof(unsigned long long));        
        unsigned long long *d_h_combined_nodes, *d_in_deg, *temp_arr;
        cudaMalloc(&d_h_combined_nodes, h_combined_nodes_s * sizeof(unsigned long long));
        cudaMemcpy(d_h_combined_nodes, h_combined_nodes, h_combined_nodes_s * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMalloc(&d_in_deg, num_nodes * sizeof(unsigned long long));
        cudaMemcpy(d_in_deg, in_deg, num_nodes * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_arr, h_combined_nodes_s * sizeof(unsigned long long));

        // Launch the kernel to create the subgraph
        unsigned long long grid_size_G = (num_nodes + BLOCK_SIZE) / BLOCK_SIZE;
        struct timeval begin4, end4;
        gettimeofday(&begin4, 0);
        Find_size<<<grid_size_G, BLOCK_SIZE>>>(d_in_deg, d_h_combined_nodes, d_row_ptr_G, d_col_idx_G, temp_arr, num_nodes, h_combined_nodes_s, 
        num_nodes+1, num_edges, d_filtered_halo_nodes_out, unique_halo_nodes_s[0], d_induced_nodes, indices_size);
        cudaDeviceSynchronize();

        void *d_temp_storage3 = NULL;
        unsigned long long* temp_arr_sum;
        cudaMalloc(&temp_arr_sum, h_combined_nodes_s * sizeof(unsigned long long));
    
        cub::DeviceScan::InclusiveSum(d_temp_storage3, temp_storage_bytes, temp_arr, temp_arr_sum, h_combined_nodes_s);
        
        // Allocate temporary storage for inclusive prefix sum
        cudaMalloc(&d_temp_storage3, temp_storage_bytes);
        
        // Run inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage2, temp_storage_bytes, temp_arr, temp_arr_sum, h_combined_nodes_s);

        // unsigned long long *temp_arr_cpu = (unsigned long long *)malloc(h_combined_nodes_s * sizeof(unsigned long long));
        // cudaMemcpy(temp_arr_cpu, temp_arr, h_combined_nodes_s * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        // for(unsigned long long i = 0; i<h_combined_nodes_s; i++){
        //     cout<<temp_arr_cpu[i]<<" ";
        // }
        // cout<<endl;

        unsigned long long *temp_arr_sum_cpu = (unsigned long long *)malloc(h_combined_nodes_s * sizeof(unsigned long long));
        cudaMemcpy(temp_arr_sum_cpu, temp_arr_sum, h_combined_nodes_s * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(temp_arr);
        cudaFree(d_temp_storage3);
        gettimeofday(&end4, 0);
        long seconds4 = end4.tv_sec - begin4.tv_sec;
        long microseconds4 = end4.tv_usec - begin4.tv_usec;
        double elapsed4 = seconds4 + microseconds4*1e-6;            
        cout<<"Time measured for find size: "<<elapsed4<<" seconds."<<endl;
        total_kernel_time += elapsed4;
        // for(unsigned long long i = 0; i<h_combined_nodes_s; i++){
        //     cout<<temp_arr_sum_cpu[i]<<" ";
        // }
        // cout<<endl;

        unsigned long long *d_row_ptr_dir, *d_col_idx_dir, *d_indegree;
        cudaMalloc(&d_row_ptr_dir, (h_combined_nodes_s+1) * sizeof(unsigned long long));
        cudaMalloc(&d_col_idx_dir, (temp_arr_sum_cpu[h_combined_nodes_s-1]) * sizeof(unsigned long long));
        cudaMalloc(&d_indegree, num_nodes*sizeof(unsigned long long));
        cudaMemset(d_indegree, 0, num_nodes*sizeof(unsigned long long));

        unsigned long long *index_array = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
        memset(index_array, 0, num_nodes * sizeof(unsigned long long));
        for(unsigned long long i = 0; i<h_combined_nodes_s; i++){
            index_array[h_combined_nodes[i]] = i;
        }

        // cout<<"Index array: "<<endl;
        // for(int i = 0; i<num_nodes; i++) cout<<index_array[i]<<" ";
        // cout<<endl;

        unsigned long long *d_index_array;
        cudaMalloc(&d_index_array, num_nodes*sizeof(unsigned long long));
        cudaMemcpy(d_index_array, index_array, num_nodes*sizeof(unsigned long long), cudaMemcpyHostToDevice);

        struct timeval begin5, end5;
        gettimeofday(&begin5, 0);
        Convert<<<grid_size_G, BLOCK_SIZE>>>(d_in_deg, d_h_combined_nodes, d_row_ptr_G, d_col_idx_G, temp_arr_sum, d_row_ptr_dir, d_col_idx_dir, 
        num_nodes, h_combined_nodes_s, h_combined_nodes_s, num_edges, d_index_array, d_indegree, d_filtered_halo_nodes_out, unique_halo_nodes_s[0],
        d_induced_nodes, indices_size);
        cudaDeviceSynchronize();

        unsigned long long int ci_pos = temp_arr_sum_cpu[h_combined_nodes_s-1];
        unsigned long long int rp_pos = (h_combined_nodes_s+1);
        unsigned long long int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
		checkCuda(cudaMalloc(&d_values_out,sizeof(unsigned long long int)*ci_pos));
		cudaMemset(d_values_out, 0, ci_pos * sizeof(unsigned long long int));
		void     *d_temp_storage = NULL;
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_col_idx_dir, d_values_out,
		    ci_pos, rp_pos-1, d_row_ptr_dir, d_row_ptr_dir + 1);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run sorting operation
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_col_idx_dir, d_values_out,
		    ci_pos, rp_pos-1, d_row_ptr_dir, d_row_ptr_dir + 1);
		// d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
		checkCuda(cudaFree(d_col_idx_dir));
		checkCuda(cudaFree(d_temp_storage));
        gettimeofday(&end5, 0);
        long seconds5 = end5.tv_sec - begin5.tv_sec;
        long microseconds5 = end5.tv_usec - begin5.tv_usec;
        double elapsed5 = seconds5 + microseconds5*1e-6;            
        cout<<"Time measured for convert undir to dir: "<<elapsed5<<" seconds."<<endl;
        total_kernel_time += elapsed5;

        unsigned long long *row_ptr_dir = (unsigned long long *)malloc((h_combined_nodes_s+1) * sizeof(unsigned long long));
        unsigned long long *col_idx_dir = (unsigned long long *)malloc((temp_arr_sum_cpu[h_combined_nodes_s-1]) * sizeof(unsigned long long));
        cudaMemcpy(row_ptr_dir, d_row_ptr_dir, (h_combined_nodes_s+1) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_idx_dir, d_values_out, (temp_arr_sum_cpu[h_combined_nodes_s-1]) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // cout<<"Row Pointer: "<<endl;
        // for(int i = 0; i<h_combined_nodes_s+1; i++){
        //     cout<<row_ptr_dir[i]<<" ";
        // }
        // cout<<endl;

        // cout<<"Column Index: "<<endl;
        // for(int i = 0; i<temp_arr_sum_cpu[h_combined_nodes_s-1]; i++){
        //     cout<<col_idx_dir[i]<<" ";
        // }
        // cout<<endl;


        unsigned long long *indegree = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
        cudaMemcpy(indegree, d_indegree, num_nodes * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // cout<<"Indegree: "<<endl;
        // for(int i = 0; i<num_nodes; i++){
        //     cout<<indegree[i]<<" ";
        // }
        // cout<<endl;


        // cudaMemcpy(degree.data(), d_result.degree, degree.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // cout << "Indegree of vertices:" << std::endl;
        // for (unsigned long long i = 0; i < h_combined_nodes_s; ++i) {
        //     cout << "Vertex " << h_combined_nodes[i] << ": " << degree[h_combined_nodes[i]] << endl;
        // }
        // unsigned long long *row_ptr_part = (unsigned long long *)malloc((h_combined_nodes_s+1) * sizeof(unsigned long long));
        //         cout<<"Indegree brought to cpu"<<endl;
        // unsigned long long *col_idx_part = (unsigned long long *)malloc(num_edges * sizeof(unsigned long long));
        //         cout<<"Indegree brought to cpu"<<endl;

        // unsigned long long row_ptr_part_size = 1;
        // unsigned long long col_idx_part_size = 0;

        // row_ptr_part[0] = 0;
        // cout<<"Base row ptr and col idx made"<<endl;

        // for (unsigned long long i = 0; i < h_combined_nodes_s; ++i) {
        //     // if (indegree[result_nodes[i]] != 0 || (indegree[result_nodes[i]] == 0 && i < induced_nodes_s)) {
        //         unsigned long long u = result_nodes[i];
        //         // cout<<"Node: "<<u<<" ";
        //         unsigned long long out = 0;
        //         // cout<<"Neighbors: ";
        //         for (unsigned long long j = result_neighbor_offsets[i]; j < result_neighbor_offsets[i + 1]; ++j) {
        //             signed long long v = result_neighbors[j];
        //             if (v != -1) {
        //                 col_idx_part[col_idx_part_size++] = v;
        //                 out++;
        //                 // cout<<v<<" ";
        //             }
        //             else break;
        //         }

        //         row_ptr_part[row_ptr_part_size] = row_ptr_part[row_ptr_part_size-1] + out;
        //         row_ptr_part_size++;
        //         // cout<<endl;
        //     // }
        // }

        // cout<<"Base row ptr and col idx made"<<endl;
        
        // // Update number of nodes and edges in the partition
        outfile <<h_combined_nodes_s <<" "<<h_combined_nodes_s+1 <<" " << temp_arr_sum_cpu[h_combined_nodes_s-1] <<" "<< indices_size<< endl;
        // cout<<"Row Pointer: "<<endl;
        for(int i = 0; i<h_combined_nodes_s+1; i++){
            outfile<<row_ptr_dir[i]<<" ";
            // cout<<row_ptr_dir[i]<<" ";
        }
        outfile<<endl;
        cout<<endl;

        // cout<<"Column Index: "<<endl;
        for(int i = 0; i<temp_arr_sum_cpu[h_combined_nodes_s-1]; i++){
            outfile<<col_idx_dir[i]<<" ";
            // cout<<col_idx_dir[i]<<" ";
        }
        outfile<<endl;
        cout<<endl;
        // for (unsigned long long i = 0; i < row_ptr_part_size; i++) {
        //     outfile << row_ptr_part[i] << " ";
        // }
        // outfile << endl;
        // for (unsigned long long i = 0; i < col_idx_part_size; i++) {
        //     outfile << col_idx_part[i] << " ";
        // }
        // outfile << endl;    
        // cout<<"Row ptr and COl idx made"<<endl;

        // Free dynamically allocated memory

        // cout<<"Row pounsigned long longer part:"<<endl;
        // for(unsigned long long i = 0; i<row_ptr_part_size; i++){ cout<<row_ptr_part[i]<<" "; }
        // cout<<endl;
        // cout<<"Col index part:"<<endl;
        // for(unsigned long long i = 0; i<col_idx_part_size; i++){ cout<<col_idx_part[i]<<" "; }
        // cout<<endl;
        // thrust::host_vector<unsigned long long> in_deg = indegree;
        // cout<<"Indegree"<<endl;
        // for(unsigned long long i = 0; i<in_deg.size(); i++){
        //     cout<<in_deg[i]<<" ";
        // }
        // cout<<endl;
         // Write to output file

        free(indices);
        free(h_halo_nodes);         
        free(filtered_halo_nodes);         
        free(h_combined_nodes);
        free(unique_halo_nodes);
        free(unique_halo_nodes_s);
        free(row_ptr_dir);
        free(col_idx_dir);
        free(indegree);
        free(index_array);
        free(temp_arr_sum_cpu);

        cudaFree(d_induced_nodes);
        cudaFree(d_row_ptr_G);
        cudaFree(d_col_idx_G);
        cudaFree(d_filtered_halo_nodes_out);
        cudaFree(d_filtered_halo_nodes_out_s);
        cudaFree(d_h_combined_nodes);
        cudaFree(d_in_deg);
        cudaFree(temp_arr_sum);
        cudaFree(d_col_idx_dir);
        cudaFree(d_row_ptr_dir);
        cudaFree(d_index_array);
        cudaFree(d_indegree);
        // cudaFree(d_indegree);
    }
    cout<<"Total Kernel Time: "<<total_kernel_time<<" seconds."<<endl;
   outfile.close();
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <nopart> <filename>" << endl;
        return 1;
    }

    // Parse command-line arguments
    unsigned long long nopart = stoi(argv[1]);
    string filename = argv[2];

    // Read the input file
    ifstream infile(filename);
    if (!infile.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return 1;
    }

    unsigned long long num_nodes, num_edges;
    infile >> num_nodes >> num_edges;

    unsigned long long num_master_nodes, row_ptr_size, col_idx_size, num_halo_nodes;
    infile >> num_master_nodes >> row_ptr_size >> col_idx_size >> num_halo_nodes;

    unsigned long long *row_ptr_G = (unsigned long long *)malloc(row_ptr_size * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < row_ptr_size; ++i) {
        infile >> row_ptr_G[i];
    }

    unsigned long long *col_idx_G = (unsigned long long *)malloc(col_idx_size * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < col_idx_size; ++i) {
        infile >> col_idx_G[i];
    }

    unsigned long long *in_deg = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < num_nodes; ++i) {
        infile >> in_deg[i];
    }
    // for(unsigned long long i = 0; i<num_nodes; i++) cout<<in_deg[i]<<" ";
    infile.close();

    string filename2 = argv[3];

    // Read the input file
    ifstream infile2(filename2);
    if (!infile2.is_open())
    {
        cerr << "Error opening file: " << filename2 << endl;
        return 1;
    }

    unsigned long long *node_parts = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < num_nodes; ++i) {
        infile2 >> node_parts[i];
    }

    infile2.close();

    string output_filename = filename2 + "_output.csr";
    // Call the function to create subgraph with halo nodes
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    create_subgraph_with_halo(nopart, node_parts, row_ptr_G, col_idx_G, in_deg, output_filename,num_nodes,num_edges);
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    cout<<"Time measured:"<<elapsed<<" seconds."<<endl;

    return 0;
}
