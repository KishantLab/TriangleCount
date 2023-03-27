#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<math.h>
#include<cub/cub.cuh>

#define NUM_VERTICES 9999999999
#define NUM_EDGES 9999999999
#define N_THREADS_PER_BLOCK 256
#define SHARED_MEM 256

//---------------------Binary Search -------------------------//
__device__ int Search (int skey , int *neb, int sizelist)
{
	int total = 0;
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
		int lo = 1;
		int hi = sizelist-1;
		int mid=0;
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
__global__ void Find_Triangle(int *d_col_index, int *d_row_ptr, int *d_vertex_arr,unsigned long long int *d_sum, int ci_pos)
{
	//int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	__shared__ int start;
	__shared__ int end;
	__shared__ int neb[SHARED_MEM];
unsigned long long	int triangle = 0;
/*
	   if(tid == 0 && bid == 1)
	   {
	   printf("\nCol_index: ");
	   for(int i = 0; i < ci_pos; i++)
	   {
	   printf(" %d",d_col_index[i]);
	   }
	   }
*/
	if(d_vertex_arr[bid]==1)
	{
		//if(threadIdx.x==0 && bid==0) printf("Hello from Kernel : %d\n",bid);

		if (tid == 0)
		{
			start = d_row_ptr[bid];
			end = d_row_ptr[bid+1]-1;
		}
		__syncthreads();
		int size_list1 = end - start;
		if(size_list1 < N_THREADS_PER_BLOCK)
		{
			if(tid <= size_list1)
			{
				neb[tid] = d_col_index[tid+start];
			}
			__syncthreads();
			for( int i = 0; i <= size_list1; i++)
			{
				int start2 = d_row_ptr[neb[i]];
				int end2 = d_row_ptr[neb[i]+1]-1;
				int size_list2 = end2 - start2;
				int M = ceil((float)(size_list2 +1)/N_THREADS_PER_BLOCK);
				for( int k = 0; k < M; k++)
				{
					int id = N_THREADS_PER_BLOCK * k + tid;
					if(id <= size_list2)
					{
						int result = 0;
						result = Search(d_col_index[id+start2],neb,size_list1);
						//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d skey:%d, neb[0]:%d ,neb[%d]:%d",bid, neb[i], result,tid,size_list1+1,size_list2+1,start2,end2,d_col_index[id+start2],neb[0],size_list1,neb[size_list1]);
						//atomicAdd(&g_sum[0],result);
						//printf("\nedge(%d , %d) src : %d dst :%d ", bid,neb[i],size_list1+1,size_list2+1);
						triangle += result;
					}
				}
			}
		}
		else
		{
			int N = ceil((float)(size_list1 +1)/ N_THREADS_PER_BLOCK);
			int remining_size = size_list1;
			int size = N_THREADS_PER_BLOCK-1;
			for( int i = 0; i < N; i++)
			{
				int id = N_THREADS_PER_BLOCK * i + tid;
				if( remining_size > size)
				{
					if(id <= size_list1)
					{
						neb[tid] = d_col_index[id+start];
						//printf(" neb : %d", neb[tid]);
					}
					__syncthreads();
					for( int j = start; j <= end; j++)
					{
						int start2 = d_row_ptr[d_col_index[j]];
						int end2 = d_row_ptr[d_col_index[j]+1]-1;
						int size_list2 = end2 - start2;
						int M = ceil((float)(size_list2 +1)/N_THREADS_PER_BLOCK);
						for( int k = 0; k < M; k++)
						{
							int tempid = N_THREADS_PER_BLOCK * k + tid;
							if(tempid <= size_list2)
							{
								int result = 0;
								result = Search(d_col_index[tempid+start2],neb,size);
								//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d, remining_size:%d, size:%d, neb[0]:%d, neb[%d]:%d if ",bid, d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,remining_size,size,neb[0],size,neb[size]);
								//atomicAdd(&g_sum[0],result);
								//printf("\nedge(%d , %d) src : %d dst :%d ", bid,g_col_index[j],size_list1+1,size_list2+1);
								triangle += result;
							}
						}
					}
					__syncthreads();
					remining_size = remining_size-(size+1);
				}
				else
				{

					if(id < size_list1)
					{
						neb[tid] = d_col_index[id+start];
						//printf(" neb : %d", neb[tid]);
					}
					__syncthreads();
					for( int j = start; j <= end; j++)
					{
						int start2 = d_row_ptr[d_col_index[j]];
						int end2 = d_row_ptr[d_col_index[j]+1]-1;
						int size_list2 = end2 - start2;
						int M = ceil((float)(size_list2 +1)/ N_THREADS_PER_BLOCK);
						for (int k = 0; k < M; k++)
						{
							int tempid = N_THREADS_PER_BLOCK * k + tid;
							if(tempid <= size_list2)
							{
								int result = 0;
								result = Search(d_col_index[tempid+start2],neb,remining_size);
								//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d neb[0]:%d, neb[%d]:%d, else",bid, d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,neb[0],remining_size,neb[remining_size]);
								//atomicAdd(&g_sum[0],result);
								//printf("\nedge(%d , %d) src : %d dst :%d ", bid,g_col_index[j],size_list1+1,size_list2+1);
								triangle += result;
							}
						}
					}
				}
				__syncthreads();
			}
		}
		//if(tid ==0)
		//printf("Block Id %d Thread Id %d triangles %d \n",bid, tid, triangle);
	}
	atomicAdd(&d_sum[0],triangle);
}
int main(int argc, char *argv[])
{
	//------initilization of variables------------//
	unsigned long long int Total_Triangle = 0;

	float total_kernel_time = 0.0 ;
	float total_time = 0.0;

	int v_pos, rp_pos, ci_pos, t_ver;

	//char *argument2 = argv[2]; //take argument from terminal and initilize
	//int vertex=atoi(argument2);

	//char *argument3 = argv[3]; //take argument from terminal and initilize
	//int edge=atoi(argument3);

	char *argument4 = argv[2];
	int no_partitions = atoi(argument4);

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
		for (int i = 0; i < no_partitions; i++)
		//for (int i = 0; i < 1; i++)
		{
			int data = 0;
			fscanf(file, "%d" , &v_pos);
			fscanf(file, "%d" , &rp_pos);
			fscanf(file, "%d" , &ci_pos);
			fscanf(file, "%d" , &t_ver);

			int *new_col_index;
			//cudaMallocHost(&new_col_index,sizeof(int)*NUM_EDGES);
			new_col_index = (int *)malloc(sizeof(int)*ci_pos);

			int *new_row_ptr;
			//cudaMallocHost(&new_row_ptr,sizeof(int)*NUM_VERTICES);
			new_row_ptr = (int *)malloc(sizeof(int)*rp_pos);
			int *vertex_arr;
			//cudaMallocHost(&vertex_arr,sizeof(int)*NUM_VERTICES);
			vertex_arr = (int *)malloc(sizeof(int)*v_pos);

			for (int j = 0 ; j < v_pos ; j++)
			{
				fscanf(file, "%d", &data);
				vertex_arr[j]=data;
			}
			//printf("Row_ptr :");
			for (int j = 0; j < rp_pos; j++)
			{
				fscanf(file, "%d", &data);
				new_row_ptr[j]=data;
				//printf(" %d",data);
			}
			for (int j = 0; j < ci_pos; j++)
			{
				fscanf(file, "%d", &data);
				new_col_index[j]=data;
			}
			printf("\nPart %d executing : ",i);
			//--------------------Launch the kernel-------------------//

			int *d_vertex_arr;  //GPU MEMORY ALOOCATION
			cudaMalloc(&d_vertex_arr,sizeof(int)*v_pos);

			int *d_row_ptr;   // GPU MEMORY ALLOCATION
			cudaMalloc(&d_row_ptr,sizeof(int)*rp_pos);

			int *d_col_index;  //GPU MEMORY ALOOCATION
			cudaMalloc(&d_col_index,sizeof(int)*ci_pos);

			//int *s_col_index;  //GPU MEMORY ALOOCATION
			//cudaMalloc(&s_col_index,sizeof(int)*NUM_EDGES);

			//cudaDeviceSynchronize();
			unsigned long long	int *d_sum;
			unsigned long long int *sum;
			sum= (unsigned long long int *)malloc(sizeof(unsigned long long int)*1);
			cudaMalloc((void**)&d_sum,sizeof(unsigned long long int)*1);
			//float total_kernel_time = 0.0 ;

			//int nblocks = ceil((float)total_v_in_partitions / BLOCKSIZE);

			//start = clock();
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);

			//--------copy data from host to device --------------//
			cudaMemcpy(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice);
			cudaMemcpy(d_row_ptr,new_row_ptr,sizeof(int)*rp_pos,cudaMemcpyHostToDevice);
			cudaMemcpy(d_vertex_arr,vertex_arr,sizeof(int)*v_pos,cudaMemcpyHostToDevice);

			//Sorting The data

			//int  num_items=ci_pos;          // e.g., 7
			//int  num_segments=rp_pos;       // e.g., 3
			//int  *d_offsets; d_row_ptr         // e.g., [0, 3, 3, 7]
			//int  *d_keys_in; s_col_index        // e.g., [8, 6, 7, 5, 3, 0, 9]
			//int  *d_keys_out; d_col_index       // e.g., [-, -, -, -, -, -, -]
			int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
			cudaMalloc(&d_values_in,sizeof(int)*ci_pos);

			int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
			cudaMalloc(&d_values_out,sizeof(int)*ci_pos);

			// Determine temporary device storage requirements
			size_t temp_storage_bytes = 0;
			void *d_temp_storage = NULL;
			cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
					d_col_index, d_col_index, d_values_in, d_values_out,
					ci_pos, rp_pos-1, d_row_ptr, d_row_ptr+1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			// Run sorting operation
			cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
					d_col_index, d_col_index, d_values_in, d_values_out,
					ci_pos, rp_pos-1, d_row_ptr, d_row_ptr+1);
			cudaDeviceSynchronize();
			//cudaMemcpy(new_col_index,d_col_index,sizeof(int)*ci_pos,cudaMemcpyDeviceToHost);
			//cudaMemcpy(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice);
			//---------------------------kernel callled------------------//

			cudaEvent_t startG, stopG;
			cudaEventCreate(&startG);
			cudaEventCreate(&stopG);

			cudaEventRecord(startG);
			Find_Triangle<<<t_ver,N_THREADS_PER_BLOCK>>>(d_col_index,d_row_ptr,d_vertex_arr,d_sum,ci_pos);
			cudaDeviceSynchronize();
			cudaEventRecord(stopG);


			// cudaEventSynchronize(stop);
			//float millisecondsG = 0;
			//cudaEventElapsedTime(&millisecondsG, startG, stopG);
			//printf("    %.4f sec",millisecondsG/1000);
			//total_kernel_time = total_kernel_time + millisecondsG/1000;

			cudaMemcpy(sum,d_sum,sizeof(unsigned long long int)*1,cudaMemcpyDeviceToHost);

			unsigned long long int Triangle = sum[0];
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);

			float millisecondsG = 0;
			cudaEventElapsedTime(&millisecondsG, startG, stopG);
			//printf("    %.4f sec",millisecondsG/1000);
			total_kernel_time = total_kernel_time + millisecondsG/1000;

			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			//printf("  %.4f sec",milliseconds/1000);
			total_time = total_time + milliseconds/1000;

			printf("\t%llu\n" , Triangle);
			Total_Triangle = Total_Triangle + Triangle ;


			free(new_row_ptr);
			free(new_col_index);
			free(vertex_arr);
			free(sum);
			cudaFree(d_row_ptr);
			cudaFree(d_col_index);
			cudaFree(d_vertex_arr);
			cudaFree(d_values_in);
			cudaFree(d_values_out);
			cudaFree(d_temp_storage);
			cudaFree(d_sum);


		}
		printf("\nTotal Triangle : %llu ",Total_Triangle );
		printf("\t Total Kernel Time : %.4f sec",total_kernel_time);
		printf("\t Total Time : %f \n\n",total_time);
	}
	return 0;
}
