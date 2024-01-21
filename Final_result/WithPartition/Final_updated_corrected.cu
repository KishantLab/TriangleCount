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
__global__ void Find_Triangle(unsigned long long int *d_col_index, unsigned long long int *d_row_ptr,unsigned long long int *d_sum, unsigned long long int ci_pos, unsigned long long int rp_pos)
{
	//int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	unsigned long long int bid=blockIdx.x;
	unsigned long long int tid=threadIdx.x;

	__shared__ unsigned long long int start;
	__shared__ unsigned long long int end;
	__shared__ unsigned long long int neb[SHARED_MEM];
	__shared__ unsigned long long int s_sum[N_THREADS_PER_BLOCK];
	unsigned long long int triangle=0;
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
	//if(threadIdx.x==0) printf("Hello from Kernel : %llu\n",bid);

	if (tid==0)
	{
		start=d_row_ptr[bid];
		end=d_row_ptr[bid+1]-1;
	}
	__syncthreads();
	unsigned long long int size_list1=end-start;
	if(size_list1 < N_THREADS_PER_BLOCK)
	{
		if(tid <= size_list1)
		{
			neb[tid] = d_col_index[tid+start];
		}
		__syncthreads();
		for(unsigned long long int i=0; i<=size_list1; i++)
		{
			unsigned long long int start2 = d_row_ptr[neb[i]];
			unsigned long long int end2 = d_row_ptr[neb[i]+1]-1;
			unsigned long long int size_list2 = end2 - start2;
			unsigned long long int M = ceil((float)(size_list2+1)/N_THREADS_PER_BLOCK);
#pragma unroll
			for( unsigned long long int k = 0; k < M; k++)
			{
				unsigned long long int id = N_THREADS_PER_BLOCK * k + tid;
				if(id <= size_list2)
				{
					unsigned long long int result = 0;
					result = Search(d_col_index[id+start2],neb,size_list1);
					//printf("\nedge(%llu , %llu) : %llu , tid : %llu, size_list1 :%llu , size_list2: %llu, start2 :%llu , end2 :%llu skey:%llu, neb[0]:%llu ,neb[%llu]:%llu",bid, neb[i], result,tid,size_list1+1,size_list2+1,start2,end2,d_col_index[id+start2],neb[0],size_list1,neb[size_list1]);
					//atomicAdd(&g_sum[0],result);
					//printf("\nedge(%llu , %llu) src : %llu dst :%llu ", bid,neb[i],size_list1+1,size_list2+1);
					triangle += result;

					//if(result>=1) printf("%llu \n", result);
				}
			}
		}
	}
	else
	{
		unsigned long long int N = ceil((float)(size_list1+1)/ N_THREADS_PER_BLOCK);
		unsigned long long int remining_size = size_list1;
		unsigned long long int size = N_THREADS_PER_BLOCK-1;
		for( unsigned long long int i = 0; i < N; i++)
		{
			unsigned long long int id = N_THREADS_PER_BLOCK * i + tid;
			if( remining_size > size)
			{
				if(id <= size_list1)
				{
					neb[tid] = d_col_index[id+start];
					//prunsigned long long intf(" neb : %d", neb[tid]);
				}
				__syncthreads();
				for( unsigned long long int j = start; j <= end; j++)
				{
					unsigned long long int start2 = d_row_ptr[d_col_index[j]];
					unsigned long long int end2 = d_row_ptr[d_col_index[j]+1]-1;
					unsigned long long int size_list2 = end2 - start2;
					unsigned long long int M = ceil((float)(size_list2+1)/N_THREADS_PER_BLOCK);
#pragma unroll
					for( unsigned long long int k = 0; k < M; k++)
					{
						unsigned long long int tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2)
						{
							unsigned long long int result = 0;
							result = Search(d_col_index[tempid+start2],neb,size);
							//printf("\nedge(%llu , %llu) : %llu , tid : %llu, size_list1 :%llu , size_list2: %llu, start2 :%llu , end2 :%llu, id :%llu, skey :%llu, N:%llu, I:%llu, remining_size:%llu, size:%llu, neb[0]:%llu, neb[%llu]:%llu if ",bid, d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,remining_size,size,neb[0],size,neb[size]);
							//atomicAdd(&g_sum[0],result);
							//printf("\nedge(%llu , %llu) src : %llu dst :%llu ", bid,d_col_index[j],size_list1+1,size_list2+1);
							triangle += result;
							//if(result>=1) printf("%llu \n", result);
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
				for( unsigned long long int j = start; j <= end; j++)
				{
					unsigned long long int start2 = d_row_ptr[d_col_index[j]];
					unsigned long long int end2 = d_row_ptr[d_col_index[j]+1]-1;
					unsigned long long int size_list2 = end2 - start2;
					unsigned long long int M = ceil((float)(size_list2+1)/ N_THREADS_PER_BLOCK);
#pragma unroll
					for (unsigned long long int k = 0; k < M; k++)
					{
						unsigned long long int tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2)
						{
							unsigned long long int result = 0;
							result = Search(d_col_index[tempid+start2],neb,remining_size);
							//printf("\nedge(%llu , %llu) : %llu , tid : %llu, size_list1 :%llu , size_list2: %llu, start2 :%llu , end2 :%llu, id :%llu, skey :%llu, N:%llu, I:%llu neb[0]:%llu, neb[%llu]:%llu, else",bid, d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,neb[0],remining_size,neb[remining_size]);
							//atomicAdd(&g_sum[0],result);
							//printf("\nedge(%llu , %llu) src : %llu dst :%llu ", bid,d_col_index[j],size_list1+1,size_list2+1);
							triangle += result;

							//if(result>=1) printf("%llu \n", result);
						}
					}
					__syncthreads();				}
			}
		}
	}
	//if(tid ==0)
	//printf("Block Id %llu Thread Id %llu triangles %d \n",bid, tid, triangle);
	//}
	//atomicAdd(&d_sum[0],triangle);
	s_sum[tid] = triangle;
	__syncthreads();
	if (tid == 0)
	{
		unsigned long long int block_sum = 0;
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
	unsigned long long int Total_Triangle=0;
	float total_kernel_time=0.0 ;
	float total_time=0.0;
	cudaSetDevice(0);

	char *argument4=argv[2];
	unsigned long long int no_partitions=atoi(argument4);
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
		unsigned long long int Vertex=0, Edges=0;
		fscanf(file, "%llu", &Vertex);
		fscanf(file, "%llu", &Edges);
		for (unsigned long long int i=0; i<no_partitions; i++)
			//	for (unsigned long long int i = 0; i < 1; i++)
		{
			unsigned long long int data=0;
			unsigned long long int rp_pos=0, ci_pos=0, t_ver=0,v_pos=0;
			unsigned long long int Triangle =0;
			fscanf(file, "%llu", &v_pos);
			//fscanf(file, "%llu", &t_ver);
			fscanf(file, "%llu", &rp_pos);
			fscanf(file, "%llu", &ci_pos);
			fscanf(file, "%llu", &t_ver);
			//printf("v_pos : %llu, ci_pos: %llu,rp_pos :%llu, t_ver: %llu",v_pos,ci_pos,rp_pos,t_ver);
			unsigned long long int *new_col_index;
			//cudaMallocHost(&new_col_index,sizeof(unsigned long long int)*NUM_EDGES);
			new_col_index = (unsigned long long int *)malloc(sizeof(unsigned long long int)*ci_pos);
			unsigned long long int *new_row_ptr;
			//cudaMallocHost(&new_row_ptr,sizeof(int)*NUM_VERTICES);
			new_row_ptr = (unsigned long long int *)malloc(sizeof(unsigned long long int)*rp_pos);
			//unsigned long long int *vertex_arr;
			//cudaMallocHost(&vertex_arr,sizeof(int)*NUM_VERTICES);
			//vertex_arr = (unsigned long long int *)malloc(sizeof(unsigned long long int)*v_pos);

			//for (unsigned long long int j=0; j<v_pos; j++)
			//{
			//fscanf(file, "%llu", &data);
			//vertex_arr[j]=data;
			//}
			//printf("Row_ptr :");
			for (unsigned long long int j=0; j<rp_pos; j++)
			{
				fscanf(file, "%llu", &data);
				new_row_ptr[j]=data;
				//prunsigned long long intf(" %d",data);
			}
			for (unsigned long long int j=0; j<ci_pos; j++)
			{
				fscanf(file, "%llu", &data);
				new_col_index[j]=data;
			}
			printf("\nPart %llu executing :",i);
			//--------------------Launch the kernel-------------------//

			//unsigned long long int *d_vertex_arr;  //GPU MEMORY ALOOCATION
			//checkCuda(cudaMalloc(&d_vertex_arr,sizeof(unsigned long long int)*v_pos));

			unsigned long long int *d_row_ptr;   // GPU MEMORY ALLOCATION
			checkCuda(cudaMalloc(&d_row_ptr,sizeof(unsigned long long int)*rp_pos));

			unsigned long long int *d_col_index;  //GPU MEMORY ALOOCATION
			checkCuda(cudaMalloc(&d_col_index,sizeof(unsigned long long int)*ci_pos));
			cudaMemset(d_row_ptr, 0, rp_pos * sizeof(unsigned long long int));
			cudaMemset(d_col_index, 0, ci_pos * sizeof(unsigned long long int));
			//int *s_col_index;  //GPU MEMORY ALOOCATION
			//cudaMalloc(&s_col_index,sizeof(int)*NUM_EDGES);

			//cudaDeviceSynchronize();

			unsigned long long int *out;
			out=(unsigned long long int *)malloc(sizeof(unsigned long long int)*1);

			unsigned long long int *d_sum;
			checkCuda(cudaMalloc(&d_sum,sizeof(unsigned long long int)*t_ver));
			cudaMemset(d_sum, 0, t_ver * sizeof(unsigned long long int));
			unsigned long long int *d_out;
			cudaMalloc((void**)&d_out,sizeof(unsigned long long int)*1);
			cudaMemset(d_out, 0, 1 * sizeof(unsigned long long int));
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
			checkCuda(cudaMemcpy(d_col_index,new_col_index,sizeof(unsigned long long int)*ci_pos,cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_row_ptr,new_row_ptr,sizeof(unsigned long long int)*rp_pos,cudaMemcpyHostToDevice));
			//checkCuda(cudaMemcpy(d_vertex_arr,vertex_arr,sizeof(unsigned long long int)*v_pos,cudaMemcpyHostToDevice));

			//Sorting The data

			//int  num_items=ci_pos;          // e.g., 7
			//int  num_segments=rp_pos;       // e.g., 3
			//int  *d_offsets; d_row_ptr         // e.g., [0, 3, 3, 7]
			//int  *d_keys_in; s_col_index        // e.g., [8, 6, 7, 5, 3, 0, 9]
			//int  *d_keys_out; d_col_index       // e.g., [-, -, -, -, -, -, -]
			// unsigned long long int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
			// checkCuda(cudaMalloc(&d_values_in,sizeof(unsigned long long int)*ci_pos));
			// cudaMemset(d_values_in, 0, ci_pos * sizeof(unsigned long long int));
			//
			// unsigned long long int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
			// checkCuda(cudaMalloc(&d_values_out,sizeof(unsigned long long int)*ci_pos));
			// cudaMemset(d_values_out, 0, ci_pos * sizeof(unsigned long long int));
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
			///commented part
			//Determine temporary device storage requirements
			unsigned long long int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
			checkCuda(cudaMalloc(&d_values_out,sizeof(unsigned long long int)*ci_pos));
			cudaMemset(d_values_out, 0, ci_pos * sizeof(unsigned long long int));

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
			///commented part
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

			//checkCuda(cudaMemcpy(sum,d_sum,sizeof(unsigned long long int)*1,cudaMemcpyDeviceToHost));

			//unsigned long long int Triangle=sum[0];
			checkCuda(cudaEventRecord(stop));
			checkCuda(cudaEventSynchronize(stop));

			checkCuda(cudaFree(d_row_ptr));
			checkCuda(cudaFree(d_col_index));
			//checkCuda(cudaFree(d_values_in));
			//checkCuda(cudaFree(d_values_out));
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
			cudaMemcpy(out,d_out,sizeof(unsigned long long int)*1,cudaMemcpyDeviceToHost);
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

			printf("\t%llu\n" , Triangle);
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
		printf("\nTotal Vertex : %llu ",Vertex );
		printf("\tTotal Edges : %llu ",Edges/2 );
		printf("\tTotal Triangle : %llu ",Total_Triangle );
		printf("\tTotal Kernel Time : %.4f sec",total_kernel_time);
		printf("\tTotal Time : %f \n\n",total_time);
	}
	return 0;
}
