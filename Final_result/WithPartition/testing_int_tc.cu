#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<math.h>
#include<cub/cub.cuh>
#include<cuda_runtime_api.h>

//#define NUM_VERTICES 9999999999
//#define NUM_EDGES 9999999999
#define N_THREADS_PER_BLOCK 128
#define SHARED_MEM 128

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

//---------------------Binary Search -------------------------//
__device__ __forceinline__ uint Search (uint skey , uint *neb, uint sizelist)
{
	uint total = 0;
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
		uint lo=1;
		uint hi=sizelist-1;
		uint mid=0;
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
__global__ void Find_Triangle(uint *d_col_index, uint *d_row_ptr,uint *d_sum, uint ci_pos, uint rp_pos)
{
	//int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	uint bid=blockIdx.x;
	uint tid=threadIdx.x;

	__shared__ uint start;
	__shared__ uint end;
	__shared__ uint neb[SHARED_MEM];
	__shared__ uint s_sum[N_THREADS_PER_BLOCK];
	uint triangle=0;
	/*
	   if(tid == 0 && bid == 1)
	   {
	   printf("Row Ptr :");
	   for(int i =0; i<rp_pos; i++)
	   {
	   printf(" %d",d_row_ptr[i]);
	   }
	   printf("\nCol_index: ");
	   for(int i = 0; i < ci_pos; i++)
	   {
	   printf(" %d",d_col_index[i]);
	   }
	   }
	 */
	//if(d_vertex_arr[bid]==1)
	//if (bid < v_pos)
	//{
	//if(threadIdx.x==0) printf("Hello from Kernel : %u\n",bid);

	if (tid==0)
	{
		start=d_row_ptr[bid];
		end=d_row_ptr[bid+1]-1;
	}
	__syncthreads();
	uint size_list1=end-start;
	if(size_list1 < N_THREADS_PER_BLOCK)
	{
		if(tid <= size_list1)
		{
			neb[tid] = d_col_index[tid+start];
		}
		__syncthreads();
		for(uint i=0; i<=size_list1; i++)
		{
			uint start2 = d_row_ptr[neb[i]];
			uint end2 = d_row_ptr[neb[i]+1]-1;
			uint size_list2 = end2 - start2;
			uint M = ceil((float)(size_list2+1)/N_THREADS_PER_BLOCK);
#pragma unroll
			for( uint k = 0; k < M; k++)
			{
				uint id = N_THREADS_PER_BLOCK * k + tid;
				if(id <= size_list2)
				{
					uint result = 0;
					result = Search(d_col_index[id+start2],neb,size_list1);
					//printf("\nedge(%u , %u) : %u , tid : %u, size_list1 :%u , size_list2: %u, start2 :%u , end2 :%u skey:%u, neb[0]:%u ,neb[%u]:%u",bid, neb[i], result,tid,size_list1+1,size_list2+1,start2,end2,d_col_index[id+start2],neb[0],size_list1,neb[size_list1]);
					//atomicAdd(&g_sum[0],result);
					//printf("\nedge(%u , %u) src : %u dst :%u ", bid,neb[i],size_list1+1,size_list2+1);
					triangle += result;

					//if(result>=1) printf("%u \n", result);
				}
			}
		}
	}
	else
	{
		uint N = ceil((float)(size_list1+1)/ N_THREADS_PER_BLOCK);
		uint remining_size = size_list1;
		uint size = N_THREADS_PER_BLOCK-1;
		for( uint i = 0; i < N; i++)
		{
			uint id = N_THREADS_PER_BLOCK * i + tid;
			if( remining_size > size)
			{
				if(id <= size_list1)
				{
					neb[tid] = d_col_index[id+start];
					//pruintf(" neb : %d", neb[tid]);
				}
				__syncthreads();
				for( uint j = start; j <= end; j++)
				{
					uint start2 = d_row_ptr[d_col_index[j]];
					uint end2 = d_row_ptr[d_col_index[j]+1]-1;
					uint size_list2 = end2 - start2;
					uint M = ceil((float)(size_list2+1)/N_THREADS_PER_BLOCK);
#pragma unroll
					for( uint k = 0; k < M; k++)
					{
						uint tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2)
						{
							uint result = 0;
							result = Search(d_col_index[tempid+start2],neb,size);
							//printf("\nedge(%u , %u) : %u , tid : %u, size_list1 :%u , size_list2: %u, start2 :%u , end2 :%u, id :%u, skey :%u, N:%u, I:%u, remining_size:%u, size:%u, neb[0]:%u, neb[%u]:%u if ",bid, d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,remining_size,size,neb[0],size,neb[size]);
							//atomicAdd(&g_sum[0],result);
							//printf("\nedge(%u , %u) src : %u dst :%u ", bid,d_col_index[j],size_list1+1,size_list2+1);
							triangle += result;
							//if(result>=1) printf("%u \n", result);
						}
					}
					__syncthreads();
				}
				__syncthreads();
				remining_size = remining_size-(size+1);
			}
			else
			{

				if(id <= size_list1)
				{
					neb[tid] = d_col_index[id+start];
					//printf(" neb : %d", neb[tid]);
				}
				__syncthreads();
				for( uint j = start; j <= end; j++)
				{
					uint start2 = d_row_ptr[d_col_index[j]];
					uint end2 = d_row_ptr[d_col_index[j]+1]-1;
					uint size_list2 = end2 - start2;
					uint M = ceil((float)(size_list2+1)/ N_THREADS_PER_BLOCK);
#pragma unroll
					for (uint k = 0; k < M; k++)
					{
						uint tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2)
						{
							uint result = 0;
							result = Search(d_col_index[tempid+start2],neb,remining_size);
							//printf("\nedge(%u , %u) : %u , tid : %u, size_list1 :%u , size_list2: %u, start2 :%u , end2 :%u, id :%u, skey :%u, N:%u, I:%u neb[0]:%u, neb[%u]:%u, else",bid, d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,neb[0],remining_size,neb[remining_size]);
							//atomicAdd(&g_sum[0],result);
							//printf("\nedge(%u , %u) src : %u dst :%u ", bid,d_col_index[j],size_list1+1,size_list2+1);
							triangle += result;

							//if(result>=1) printf("%u \n", result);
						}
					}
					__syncthreads();				}
			}
		}
	}
	//if(tid ==0)
	//printf("Block Id %u Thread Id %u triangles %d \n",bid, tid, triangle);
	//}
	//atomicAdd(&d_sum[0],triangle);
	s_sum[tid] = triangle;
	__syncthreads();
	if (tid == 0)
	{
		uint block_sum = 0;
		for (int i = 0; i < N_THREADS_PER_BLOCK; i++)
		{
			block_sum += s_sum[i];
		}
		__syncthreads();
		d_sum[bid] = block_sum;
	}
}
int main(int argc, char *argv[])
{
	//------initilization of variables------------//
	uint Total_Triangle=0;
	float total_kernel_time=0.0 ;
	float total_time=0.0;
	cudaSetDevice(0);

	char *argument4=argv[2];
	uint no_partitions=atoi(argument4);
	//--------------------Load DATA In Memory---------------------//

	FILE *file;
	file = fopen(argv[1],"r");
	if (file == NULL)
	{
		printf("\nFile Not Opened.........");
		exit(0);
	}
	else
	{
		uint Vertex=0, Edges=0;
		fscanf(file, "%u", &Vertex);
		fscanf(file, "%u", &Edges);
		for (uint i=0; i<no_partitions; i++)
			//	for (uint i = 0; i < 1; i++)
		{
			uint data=0;
			uint rp_pos=0, ci_pos=0, t_ver=0,v_pos=0;
			uint Triangle =0;
			fscanf(file, "%u", &v_pos);
			//fscanf(file, "%u", &t_ver);
			fscanf(file, "%u", &rp_pos);
			fscanf(file, "%u", &ci_pos);
			fscanf(file, "%u", &t_ver);
			//printf("v_pos : %u, ci_pos: %u,rp_pos :%u, t_ver: %u",v_pos,ci_pos,rp_pos,t_ver);
			uint *new_col_index;
			//cudaMallocHost(&new_col_index,sizeof(uint)*NUM_EDGES);
			new_col_index = (uint *)malloc(sizeof(uint)*ci_pos);
			uint *new_row_ptr;
			//cudaMallocHost(&new_row_ptr,sizeof(int)*NUM_VERTICES);
			new_row_ptr = (uint *)malloc(sizeof(uint)*rp_pos);
			//uint *vertex_arr;
			//cudaMallocHost(&vertex_arr,sizeof(int)*NUM_VERTICES);
			//vertex_arr = (uint *)malloc(sizeof(uint)*v_pos);

			//for (uint j=0; j<v_pos; j++)
			//{
			//fscanf(file, "%u", &data);
			//vertex_arr[j]=data;
			//}
			//printf("Row_ptr :");
			for (uint j=0; j<rp_pos; j++)
			{
				fscanf(file, "%u", &data);
				new_row_ptr[j]=data;
				//pruintf(" %d",data);
			}
			for (uint j=0; j<ci_pos; j++)
			{
				fscanf(file, "%u", &data);
				new_col_index[j]=data;
			}
			printf("\nPart %u executing :",i);
			//--------------------Launch the kernel-------------------//

			//uint *d_vertex_arr;  //GPU MEMORY ALOOCATION
			//checkCuda(cudaMalloc(&d_vertex_arr,sizeof(uint)*v_pos));

			uint *d_row_ptr;   // GPU MEMORY ALLOCATION
			checkCuda(cudaMalloc(&d_row_ptr,sizeof(uint)*rp_pos));

			uint *d_col_index;  //GPU MEMORY ALOOCATION
			checkCuda(cudaMalloc(&d_col_index,sizeof(uint)*ci_pos));
			cudaMemset(d_row_ptr, 0, rp_pos * sizeof(uint));
			cudaMemset(d_col_index, 0, ci_pos * sizeof(uint));
			//int *s_col_index;  //GPU MEMORY ALOOCATION
			//cudaMalloc(&s_col_index,sizeof(int)*NUM_EDGES);

			//cudaDeviceSynchronize();

			uint *out;
			out=(uint *)malloc(sizeof(uint)*1);

			uint *d_sum;
			checkCuda(cudaMalloc(&d_sum,sizeof(uint)*t_ver));
			cudaMemset(d_sum, 0, t_ver * sizeof(uint));
			uint *d_out;
			cudaMalloc((void**)&d_out,sizeof(uint)*1);
			cudaMemset(d_out, 0, 1 * sizeof(uint));
			//float total_kernel_time = 0.0 ;

			//int nblocks = ceil((float)total_v_in_partitions / BLOCKSIZE);

			//start = clock();
			cudaEvent_t start, stop;
			checkCuda( cudaEventCreate(&start));
			checkCuda( cudaEventCreate(&stop));
			cudaEventRecord(start);

			cudaEvent_t start_sum,stop_sum;
			cudaEventCreate(&start_sum);
			cudaEventCreate(&stop_sum);
			//--------copy data from host to device --------------//
			checkCuda(cudaMemcpy(d_col_index,new_col_index,sizeof(uint)*ci_pos,cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_row_ptr,new_row_ptr,sizeof(uint)*rp_pos,cudaMemcpyHostToDevice));
			//checkCuda(cudaMemcpy(d_vertex_arr,vertex_arr,sizeof(uint)*v_pos,cudaMemcpyHostToDevice));

			//Sorting The data

			//int  num_items=ci_pos;          // e.g., 7
			//int  num_segments=rp_pos;       // e.g., 3
			//int  *d_offsets; d_row_ptr         // e.g., [0, 3, 3, 7]
			//int  *d_keys_in; s_col_index        // e.g., [8, 6, 7, 5, 3, 0, 9]
			//int  *d_keys_out; d_col_index       // e.g., [-, -, -, -, -, -, -]
			// uint  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
			// checkCuda(cudaMalloc(&d_values_in,sizeof(uint)*ci_pos));
			// cudaMemset(d_values_in, 0, ci_pos * sizeof(uint));
			//
			// uint  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
			// checkCuda(cudaMalloc(&d_values_out,sizeof(uint)*ci_pos));
			// cudaMemset(d_values_out, 0, ci_pos * sizeof(uint));
			//
			//
			// // Determine temporary device storage requirements
			// size_t temp_storage_bytes=0;
			// void *d_temp_storage=NULL;
			// cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
			// 		d_col_index, d_col_index, d_values_in, d_values_out,
			// 		ci_pos, rp_pos-1, d_row_ptr, d_row_ptr+1);
			// // Allocate temporary storage
			// checkCuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			// // Run sorting operation
			// cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
			// 		d_col_index, d_col_index, d_values_in, d_values_out,
			// 		ci_pos, rp_pos-1, d_row_ptr, d_row_ptr+1);
			// checkCuda(cudaDeviceSynchronize());

			// Declare, allocate, and initialize device-accessible pointers for sorting data
			// int  num_items;          // e.g., 7
			// int  num_segments;       // e.g., 3
			// int  *d_offsets;         // e.g., [0, 3, 3, 7]
			// int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
			// int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
			// ...
			// Determine temporary device storage requirements
			uint  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
			checkCuda(cudaMalloc(&d_values_out,sizeof(uint)*ci_pos));
			cudaMemset(d_values_out, 0, ci_pos * sizeof(uint));

			void     *d_temp_storage = NULL;
			size_t   temp_storage_bytes = 0;
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_col_index, d_values_out,
			    ci_pos, rp_pos-1, d_row_ptr, d_row_ptr + 1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			// Run sorting operation
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_col_index, d_values_out,
			    ci_pos, rp_pos-1, d_row_ptr, d_row_ptr + 1);
			// d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
			checkCuda(cudaFree(d_col_index));
			checkCuda(cudaFree(d_temp_storage));
			//cudaMemcpy(new_col_index,d_col_index,sizeof(int)*ci_pos,cudaMemcpyDeviceToHost);
			//cudaMemcpy(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice);
			//---------------------------kernel callled------------------//
			checkCuda(cudaGetLastError());
			cudaEvent_t startG, stopG;
			checkCuda(cudaEventCreate(&startG));
			checkCuda(cudaEventCreate(&stopG));

			cudaEventRecord(startG);
			Find_Triangle<<<t_ver,N_THREADS_PER_BLOCK>>>(d_values_out,d_row_ptr,d_sum,ci_pos,rp_pos);
			checkCuda(cudaDeviceSynchronize());
			checkCuda(cudaEventRecord(stopG));
			checkCuda(cudaGetLastError());

			//checkCuda(cudaMemcpy(sum,d_sum,sizeof(uint)*1,cudaMemcpyDeviceToHost));

			//uint Triangle=sum[0];
			checkCuda(cudaEventRecord(stop));
			checkCuda(cudaEventSynchronize(stop));

			checkCuda(cudaFree(d_row_ptr));
			//checkCuda(cudaFree(d_col_index));
			//checkCuda(cudaFree(d_values_in));
			checkCuda(cudaFree(d_values_out));
			//checkCuda(cudaFree(d_temp_storage));
			checkCuda(cudaDeviceSynchronize());
			cudaEventRecord(start_sum);
			//---------------------perform CUMSUM----------------------//
			void *d_temp_storage_1 = NULL;
			size_t temp_storage_bytes_1 = 0;
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage_1, temp_storage_bytes_1);
			// Run sum-reduction
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);
			// d_out <-- [38]
			cudaEventRecord(stop_sum);
			checkCuda(cudaDeviceSynchronize());
			checkCuda(cudaGetLastError());
			cudaMemcpy(out,d_out,sizeof(uint)*1,cudaMemcpyDeviceToHost);
			Triangle = out[0];
			cudaEventSynchronize(stop_sum);

			float milliseconds_sum = 0;
			cudaEventElapsedTime(&milliseconds_sum, start_sum, stop_sum);
			float millisecondsG=0;
			checkCuda(cudaEventElapsedTime(&millisecondsG, startG, stopG));
			//printf("    %.4f sec",millisecondsG/1000);
			total_kernel_time = total_kernel_time + (millisecondsG/1000) + (milliseconds_sum/1000) ;

			float milliseconds=0;
			checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
			//printf("  %.4f sec",milliseconds/1000);
			total_time = total_time + milliseconds/1000;

			printf("\t%u\n" , Triangle);
			Total_Triangle = Total_Triangle + Triangle ;


			free(new_row_ptr);
			free(new_col_index);
			//free(vertex_arr);
			//free(sum);
			//checkCuda(cudaFree(d_vertex_arr));
			checkCuda(cudaFree(d_temp_storage_1));
			checkCuda(cudaFree(d_sum));
			checkCuda(cudaFree(d_out));


		}
		printf("\nTotal Vertex : %u ",Vertex );
		printf("\tTotal Edges : %u ",Edges/2 );
		printf("\tTotal Triangle : %u ",Total_Triangle );
		printf("\tTotal Kernel Time : %.4f sec",total_kernel_time);
		printf("\tTotal Time : %f \n\n",total_time);
	}
	return 0;
}
