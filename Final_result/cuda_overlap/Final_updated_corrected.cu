#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<math.h>
#include<cub/cub.cuh>
#include<cuda_runtime_api.h>

//#define NUM_VERTICES 9999999999
//#define NUM_EDGES 9999999999
#define N_THREADS_PER_BLOCK 256
#define SHARED_MEM 256

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
		//for (unsigned long long int i=0; i<no_partitions; i++)
			//	for (unsigned long long int i = 0; i < 1; i++)
		//{
		//unsigned long long int rp_pos=0, ci_pos=0, t_ver=0,v_pos=0;
		//unsigned long long int Triangle =0;
			unsigned long long int data=0, N=4;
			unsigned long long int *rp_pos;
			rp_pos = (unsigned long long int *)malloc(sizeof(unsigned long long int)*N);
			unsigned long long int *ci_pos;
			ci_pos = (unsigned long long int *)malloc(sizeof(unsigned long long int)*N);
			unsigned long long int *t_ver;
			t_ver = (unsigned long long int *)malloc(sizeof(unsigned long long int)*N);
			unsigned long long int *v_pos;
			v_pos = (unsigned long long int *)malloc(sizeof(unsigned long long int)*N);

			fscanf(file, "%llu", &v_pos[0]);
			fscanf(file, "%llu", &rp_pos[0]);
			fscanf(file, "%llu", &ci_pos[0]);
			fscanf(file, "%llu", &t_ver[0]);
			//printf("v_pos : %llu, ci_pos: %llu,rp_pos :%llu, t_ver: %llu",v_pos,ci_pos,rp_pos,t_ver);
			unsigned long long int *new_col_index_0;
			new_col_index_0 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*ci_pos[0]);
			unsigned long long int *new_row_ptr_0;
			new_row_ptr_0 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*rp_pos[0]);
			for (unsigned long long int j=0; j<rp_pos[0]; j++)
			{
				fscanf(file, "%llu", &data);
				new_row_ptr_0[j]=data;
			}
			for (unsigned long long int j=0; j<ci_pos[0]; j++)
			{
				fscanf(file, "%llu", &data);
				new_col_index_0[j]=data;
			}

			fscanf(file, "%llu", &v_pos[1]);
			fscanf(file, "%llu", &rp_pos[1]);
			fscanf(file, "%llu", &ci_pos[1]);
			fscanf(file, "%llu", &t_ver[1]);
			//printf("v_pos : %llu, ci_pos: %llu,rp_pos :%llu, t_ver: %llu",v_pos,ci_pos,rp_pos,t_ver);
			unsigned long long int *new_col_index_1;
			new_col_index_1 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*ci_pos[1]);
			unsigned long long int *new_row_ptr_1;
			new_row_ptr_1 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*rp_pos[1]);
			for (unsigned long long int j=0; j<rp_pos[1]; j++)
			{
				fscanf(file, "%llu", &data);
				new_row_ptr_1[j]=data;
			}
			for (unsigned long long int j=0; j<ci_pos[1]; j++)
			{
				fscanf(file, "%llu", &data);
				new_col_index_1[j]=data;
			}

			fscanf(file, "%llu", &v_pos[2]);
			fscanf(file, "%llu", &rp_pos[2]);
			fscanf(file, "%llu", &ci_pos[2]);
			fscanf(file, "%llu", &t_ver[2]);
			//printf("v_pos : %llu, ci_pos: %llu,rp_pos :%llu, t_ver: %llu",v_pos,ci_pos,rp_pos,t_ver);
			unsigned long long int *new_col_index_2;
			new_col_index_2 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*ci_pos[2]);
			unsigned long long int *new_row_ptr_2;
			new_row_ptr_2 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*rp_pos[2]);
			for (unsigned long long int j=0; j<rp_pos[2]; j++)
			{
				fscanf(file, "%llu", &data);
				new_row_ptr_2[j]=data;
			}
			for (unsigned long long int j=0; j<ci_pos[2]; j++)
			{
				fscanf(file, "%llu", &data);
				new_col_index_2[j]=data;
			}

			fscanf(file, "%llu", &v_pos[3]);
			fscanf(file, "%llu", &rp_pos[3]);
			fscanf(file, "%llu", &ci_pos[3]);
			fscanf(file, "%llu", &t_ver[3]);
			//printf("v_pos : %llu, ci_pos: %llu,rp_pos :%llu, t_ver: %llu",v_pos,ci_pos,rp_pos,t_ver);
			unsigned long long int *new_col_index_3;
			new_col_index_3 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*ci_pos[3]);
			unsigned long long int *new_row_ptr_3;
			new_row_ptr_3 = (unsigned long long int *)malloc(sizeof(unsigned long long int)*rp_pos[3]);
			for (unsigned long long int j=0; j<rp_pos[3]; j++)
			{
				fscanf(file, "%llu", &data);
				new_row_ptr_3[j]=data;
			}
			for (unsigned long long int j=0; j<ci_pos[3]; j++)
			{
				fscanf(file, "%llu", &data);
				new_col_index_3[j]=data;
			}

			printf("\nPart %llu executing :",i);
			//--------------------Launch the kernel-------------------//

			//unsigned long long int *d_vertex_arr;  //GPU MEMORY ALOOCATION
			//checkCuda(cudaMalloc(&d_vertex_arr,sizeof(unsigned long long int)*v_pos));
			//int *s_col_index;  //GPU MEMORY ALOOCATION
			//cudaMalloc(&s_col_index,sizeof(int)*NUM_EDGES);

			//cudaDeviceSynchronize();

			unsigned long long int *d_row_ptr_0;   // GPU MEMORY ALLOCATION
			checkCuda(cudaMalloc(&d_row_ptr_0,sizeof(unsigned long long int)*rp_pos[0]));
			unsigned long long int *d_col_index_0;  //GPU MEMORY ALOOCATION
			checkCuda(cudaMalloc(&d_col_index_0,sizeof(unsigned long long int)*ci_pos[0]));
			cudaMemset(d_row_ptr_0, 0, rp_pos[0] * sizeof(unsigned long long int));
			cudaMemset(d_col_index_0, 0, ci_pos[0] * sizeof(unsigned long long int));
			unsigned long long int *out_0;
			out_0=(unsigned long long int *)malloc(sizeof(unsigned long long int)*1);
			unsigned long long int *d_sum_0;
			checkCuda(cudaMalloc(&d_sum_0,sizeof(unsigned long long int)*t_ver[0]));
			cudaMemset(d_sum_0, 0, t_ver[0] * sizeof(unsigned long long int));
			unsigned long long int *d_out_0;
			cudaMalloc((void**)&d_out_0,sizeof(unsigned long long int)*1);
			cudaMemset(d_out_0, 0, 1 * sizeof(unsigned long long int));

			unsigned long long int *d_row_ptr_1;   // GPU MEMORY ALLOCATION
			checkCuda(cudaMalloc(&d_row_ptr_1,sizeof(unsigned long long int)*rp_pos[1]));
			unsigned long long int *d_col_index_1;  //GPU MEMORY ALOOCATION
			checkCuda(cudaMalloc(&d_col_index_1,sizeof(unsigned long long int)*ci_pos[1]));
			cudaMemset(d_row_ptr_1, 0, rp_pos[1] * sizeof(unsigned long long int));
			cudaMemset(d_col_index_1, 0, ci_pos[1] * sizeof(unsigned long long int));
			unsigned long long int *out_1;
			out_1=(unsigned long long int *)malloc(sizeof(unsigned long long int)*1);
			unsigned long long int *d_sum_1;
			checkCuda(cudaMalloc(&d_sum_1,sizeof(unsigned long long int)*t_ver[1]));
			cudaMemset(d_sum_1, 0, t_ver[1] * sizeof(unsigned long long int));
			unsigned long long int *d_out_1;
			cudaMalloc((void**)&d_out_1,sizeof(unsigned long long int)*1);
			cudaMemset(d_out_1, 0, 1 * sizeof(unsigned long long int));

			unsigned long long int *d_row_ptr_2;   // GPU MEMORY ALLOCATION
			checkCuda(cudaMalloc(&d_row_ptr_2,sizeof(unsigned long long int)*rp_pos[2]));
			unsigned long long int *d_col_index_2;  //GPU MEMORY ALOOCATION
			checkCuda(cudaMalloc(&d_col_index_2,sizeof(unsigned long long int)*ci_pos[2]));
			cudaMemset(d_row_ptr_2, 0, rp_pos[2] * sizeof(unsigned long long int));
			cudaMemset(d_col_index_2, 0, ci_pos[2] * sizeof(unsigned long long int));
			unsigned long long int *out_2;
			out_2=(unsigned long long int *)malloc(sizeof(unsigned long long int)*1);
			unsigned long long int *d_sum_2;
			checkCuda(cudaMalloc(&d_sum_2,sizeof(unsigned long long int)*t_ver[2]));
			cudaMemset(d_sum_2, 0, t_ver[2] * sizeof(unsigned long long int));
			unsigned long long int *d_out_2;
			cudaMalloc((void**)&d_out_2,sizeof(unsigned long long int)*1);
			cudaMemset(d_out_2, 0, 1 * sizeof(unsigned long long int));

			unsigned long long int *d_row_ptr_3;   // GPU MEMORY ALLOCATION
			checkCuda(cudaMalloc(&d_row_ptr_3,sizeof(unsigned long long int)*rp_pos[3]));
			unsigned long long int *d_col_index_3;  //GPU MEMORY ALOOCATION
			checkCuda(cudaMalloc(&d_col_index_3,sizeof(unsigned long long int)*ci_pos[3]));
			cudaMemset(d_row_ptr_3, 0, rp_pos[3] * sizeof(unsigned long long int));
			cudaMemset(d_col_index_3, 0, ci_pos[3] * sizeof(unsigned long long int));
			unsigned long long int *out_3;
			out_3=(unsigned long long int *)malloc(sizeof(unsigned long long int)*1);
			unsigned long long int *d_sum_3;
			checkCuda(cudaMalloc(&d_sum_3,sizeof(unsigned long long int)*t_ver[3]));
			cudaMemset(d_sum_3, 0, t_ver[3] * sizeof(unsigned long long int));
			unsigned long long int *d_out_3;
			cudaMalloc((void**)&d_out_3,sizeof(unsigned long long int)*1);
			cudaMemset(d_out_3, 0, 1 * sizeof(unsigned long long int));


			cudaStream_t stream[N];
			for (int i = 0; i < N; ++i)
    	checkCuda( cudaStreamCreate(&stream[i]) );

			cudaEvent_t start, stop;
			checkCuda( cudaEventCreate(&start));
			checkCuda( cudaEventCreate(&stop));
			cudaEventRecord(start);

			cudaEvent_t start_sum,stop_sum;
			cudaEventCreate(&start_sum);
			cudaEventCreate(&stop_sum);

			//checkCuda( cudaEventRecord(startEvent,0) );
			//--------copy data from host to device --------------//
			//checkCuda(cudaMemcpy(d_vertex_arr,vertex_arr,sizeof(unsigned long long int)*v_pos,cudaMemcpyHostToDevice));

			checkCuda(cudaMemcpyAsync(d_col_index_0,new_col_index_0,sizeof(unsigned long long int)*ci_pos[0],cudaMemcpyHostToDevice,stream[0])));
			checkCuda(cudaMemcpyAsync(d_row_ptr_0,new_row_ptr_0,sizeof(unsigned long long int)*rp_pos[0],cudaMemcpyHostToDevice,stream[0])));

			checkCuda(cudaMemcpyAsync(d_col_index_1,new_col_index_1,sizeof(unsigned long long int)*ci_pos[1],cudaMemcpyHostToDevice,stream[1])));
			checkCuda(cudaMemcpyAsync(d_row_ptr_1,new_row_ptr_1,sizeof(unsigned long long int)*rp_pos[1],cudaMemcpyHostToDevice,stream[1])));

			checkCuda(cudaMemcpyAsync(d_col_index_2,new_col_index_2,sizeof(unsigned long long int)*ci_pos[2],cudaMemcpyHostToDevice,stream[2])));
			checkCuda(cudaMemcpyAsync(d_row_ptr_2,new_row_ptr_2,sizeof(unsigned long long int)*rp_pos[2],cudaMemcpyHostToDevice,stream[2])));

			checkCuda(cudaMemcpyAsync(d_col_index_3,new_col_index_3,sizeof(unsigned long long int)*ci_pos[3],cudaMemcpyHostToDevice,stream[3])));
			checkCuda(cudaMemcpyAsync(d_row_ptr_3,new_row_ptr_3,sizeof(unsigned long long int)*rp_pos[3],cudaMemcpyHostToDevice,stream[3])));

			//--------------------------------------------SHorting of new_col_index-----------------------------------//
			// Determine temporary device storage requirements
			unsigned long long int  *d_values_out_0;      // e.g., [-, -, -, -, -, -, -]
			checkCuda(cudaMalloc(&d_values_out_0,sizeof(unsigned long long int)*ci_pos[0]));
			cudaMemset(d_values_out_0, 0, ci_pos[0] * sizeof(unsigned long long int));

			void     *d_temp_storage_0 = NULL;
			size_t   temp_storage_bytes_0 = 0;
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_0, temp_storage_bytes_0, d_col_index_0, d_values_out_0, ci_pos[0], rp_pos[0]-1, d_row_ptr_0, d_row_ptr_0 + 1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage_0, temp_storage_bytes_0);
			// Run sorting operation
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_0, temp_storage_bytes_0, d_col_index_0, d_values_out_0,  ci_pos[0], rp_pos[0]-1, d_row_ptr_0, d_row_ptr_0 + 1);
			// d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
			checkCuda(cudaFree(d_col_index_0));

			unsigned long long int  *d_values_out_1;      // e.g., [-, -, -, -, -, -, -]
			checkCuda(cudaMalloc(&d_values_out_1,sizeof(unsigned long long int)*ci_pos[1]));
			cudaMemset(d_values_out_1, 0, ci_pos[1] * sizeof(unsigned long long int));

			void     *d_temp_storage_1 = NULL;
			size_t   temp_storage_bytes_1 = 0;
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_1, temp_storage_bytes_1, d_col_index_1, d_values_out_1, ci_pos[1], rp_pos[1]-1, d_row_ptr_1, d_row_ptr_1 + 1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage_1, temp_storage_bytes_1);
			// Run sorting operation
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_1, temp_storage_bytes_1, d_col_index_1, d_values_out_1,  ci_pos[1], rp_pos[1]-1, d_row_ptr_1, d_row_ptr_1 + 1);
			// d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
			checkCuda(cudaFree(d_col_index_1));

			unsigned long long int  *d_values_out_2;      // e.g., [-, -, -, -, -, -, -]
			checkCuda(cudaMalloc(&d_values_out_2,sizeof(unsigned long long int)*ci_pos[2]));
			cudaMemset(d_values_out_2, 0, ci_pos[2] * sizeof(unsigned long long int));

			void     *d_temp_storage_2 = NULL;
			size_t   temp_storage_bytes_2 = 0;
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_2, temp_storage_bytes_2, d_col_index_2, d_values_out_2, ci_pos[2], rp_pos[2]-1, d_row_ptr_2, d_row_ptr_2 + 1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage_2, temp_storage_bytes_2);
			// Run sorting operation
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_2, temp_storage_bytes_2, d_col_index_2, d_values_out_2,  ci_pos[2], rp_pos[2]-1, d_row_ptr_2, d_row_ptr_2 + 1);
			// d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
			checkCuda(cudaFree(d_col_index_2));

			unsigned long long int  *d_values_out_3;      // e.g., [-, -, -, -, -, -, -]
			checkCuda(cudaMalloc(&d_values_out_3,sizeof(unsigned long long int)*ci_pos[3]));
			cudaMemset(d_values_out_3, 0, ci_pos[3] * sizeof(unsigned long long int));

			void     *d_temp_storage_3 = NULL;
			size_t   temp_storage_bytes_3 = 0;
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_3, temp_storage_bytes_3, d_col_index_3, d_values_out_3, ci_pos[3], rp_pos[3]-1, d_row_ptr_3, d_row_ptr_3 + 1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage_3, temp_storage_bytes_3);
			// Run sorting operation
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage_3, temp_storage_bytes_3, d_col_index_3, d_values_out_3,  ci_pos[3], rp_pos[3]-1, d_row_ptr_3, d_row_ptr_3 + 1);
			// d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
			checkCuda(cudaFree(d_col_index_3));
			//cudaMemcpy(new_col_index,d_col_index,sizeof(int)*ci_pos,cudaMemcpyDeviceToHost);
			//cudaMemcpy(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice);
			//---------------------------kernel callled------------------//
			checkCuda(cudaGetLastError());
			cudaEvent_t startG, stopG;
			checkCuda(cudaEventCreate(&startG));
			checkCuda(cudaEventCreate(&stopG));

			cudaEventRecord(startG);
			Find_Triangle<<<t_ver[0], N_THREADS_PER_BLOCK, 0, stream[0]>>>(d_values_out_0, d_row_ptr_0, d_sum_0, ci_pos[0], rp_pos[0]);
			Find_Triangle<<<t_ver[1], N_THREADS_PER_BLOCK, 0, stream[1]>>>(d_values_out_1, d_row_ptr_1, d_sum_1, ci_pos[1], rp_pos[1]);
			Find_Triangle<<<t_ver[2], N_THREADS_PER_BLOCK, 0, stream[2]>>>(d_values_out_2, d_row_ptr_2, d_sum_2, ci_pos[2], rp_pos[2]);
			Find_Triangle<<<t_ver[3], N_THREADS_PER_BLOCK, 0, stream[3]>>>(d_values_out_3, d_row_ptr_3, d_sum_3, ci_pos[3], rp_pos[3]);
			//checkCuda(cudaDeviceSynchronize());
			checkCuda(cudaStreamSynchronize(stream[0]));
			checkCuda(cudaStreamSynchronize(stream[1]));
			checkCuda(cudaStreamSynchronize(stream[2]));
			checkCuda(cudaStreamSynchronize(stream[3]));
			checkCuda(cudaEventRecord(stopG));
			checkCuda(cudaGetLastError());

				//checkCuda(cudaMemcpy(sum,d_sum,sizeof(unsigned long long int)*1,cudaMemcpyDeviceToHost));

			//unsigned long long int Triangle=sum[0];
			checkCuda(cudaEventRecord(stop));
			checkCuda(cudaEventSynchronize(stop));
			//checkCuda(cudaFree(d_col_index));
			//checkCuda(cudaFree(d_values_in));

			checkCuda(cudaFree(d_row_ptr_0));
			checkCuda(cudaFree(d_values_out_0));
			checkCuda(cudaFree(d_temp_storage_0));

			checkCuda(cudaFree(d_row_ptr_1));
			checkCuda(cudaFree(d_values_out_1));
			checkCuda(cudaFree(d_temp_storage_1));

			checkCuda(cudaFree(d_row_ptr_2));
			checkCuda(cudaFree(d_values_out_2));
			checkCuda(cudaFree(d_temp_storage_2));

			checkCuda(cudaFree(d_row_ptr_3));
			checkCuda(cudaFree(d_values_out_3));
			checkCuda(cudaFree(d_temp_storage_3));

			checkCuda(cudaDeviceSynchronize());
			cudaEventRecord(start_sum);
			//---------------------perform CUMSUM----------------------//
			void *d_temp_storage_1 = NULL;
			size_t temp_storage_bytes_1 = 0;
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);
			cudaMalloc(&d_temp_storage_1, temp_storage_bytes_1);
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);

			void *d_temp_storage_1 = NULL;
			size_t temp_storage_bytes_1 = 0;
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);
			cudaMalloc(&d_temp_storage_1, temp_storage_bytes_1);
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);

			void *d_temp_storage_1 = NULL;
			size_t temp_storage_bytes_1 = 0;
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);
			cudaMalloc(&d_temp_storage_1, temp_storage_bytes_1);
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);

			void *d_temp_storage_1 = NULL;
			size_t temp_storage_bytes_1 = 0;
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);
			cudaMalloc(&d_temp_storage_1, temp_storage_bytes_1);
			cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, d_sum, d_out, t_ver);

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
