#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<time.h>

#define NUM_VERTICES 999999999
#define NUM_EDGES 999999999
#define N_THREADS_PER_BLOCK 64

//-------------------intersection function ----------------------------------
__device__ int Intersection(int src, int dst, int *g_col_index , int *g_row_ptr ,int *neb, int size_list)
{
	int total = 0 ;
	int list1_start = 0;
	int list1_end = size_list;
	int list2_start = g_row_ptr[dst];
	int list2_end = g_row_ptr[dst+1];

	while (list1_start <= list1_end && list2_start < list2_end)
	{
		if (neb[list1_start] < g_col_index[list2_start]) list1_start++ ;
		else if (g_col_index[list2_start] < neb[list1_start]) list2_start++ ;
		else if (neb[list1_start] == g_col_index[list2_start])
		{
			total++;
			list1_start++;
			list2_start++;
		}
	}
	return total; //return total triangles found by each thread...
}

__global__ void Find_Triangle(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
{
	//int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int start = g_row_ptr[bid];
	int end = g_row_ptr[bid+1]-1;
	int size_list = end - start;

	if(size_list < N_THREADS_PER_BLOCK)
	{
		__shared__ int neb[N_THREADS_PER_BLOCK];
		if(tid <= size_list)
		{
			neb[tid] = g_col_index[tid+start];
		}
		__syncthreads();
		if(tid <= size_list)
		{
			int total = 0;
			total = Intersection(bid, neb[tid], g_col_index, g_row_ptr,neb,size_list);
			//printf("\nedge(%d , %d) : %d",bid, neb[tid], total);
			atomicAdd(&g_sum[0],total);
		}
	}
	//else if(size_list > N_THREADS_PER_BLOCK && size_list <= N_THREADS_PER_BLOCK * 8)
	else
	{
		__shared__ int neb[2048];
		int N = ceil((float)size_list / N_THREADS_PER_BLOCK);
		printf("size_list :%d , N :%d\n",size_list,N);
		for(int i = 0; i < N ; i++)
		{
			int id = N_THREADS_PER_BLOCK * i + tid;
			if( id <= size_list)
			{
				neb[id] = g_col_index[id + start];
			}
		}
		__syncthreads();
		for (int i = 0; i < N; i++)
		{
			int id = N_THREADS_PER_BLOCK * i + tid;
			if( id <= size_list)
			{
				int total = 0;
				total = Intersection(bid, neb[tid], g_col_index, g_row_ptr,neb,size_list);
				//printf("\nedge(%d , %d) : %d",bid, neb[tid], total);
				atomicAdd(&g_sum[0],total);
			}
		}
		__syncthreads();
	}/*
	else if (size_list > N_THREADS_PER_BLOCK *8 && size_list <= N_THREADS_PER_BLOCK * 16)
	{
		__shared__ int neb[1024];
		int N = ceil((float)size_list / N_THREADS_PER_BLOCK);
		//printf("size_list :%d , N :%d\n",size_list,N);
		for(int i = 0; i < N ; i++)
		{
			int id = N_THREADS_PER_BLOCK * i + tid;
			if( id <= size_list)
			{
				neb[id] = g_col_index[id + start];
			}
			__syncthreads();
			if( id <= size_list)
			{
				int total = 0;
				total = Intersection(bid, neb[tid], g_col_index, g_row_ptr,neb,size_list);
				//printf("\nedge(%d , %d) : %d",bid, neb[tid], total);
				atomicAdd(&g_sum[0],total);
			}
		}
	}
	else
	{
		__shared__ int neb[2048];
		int N = ceil((float)size_list / N_THREADS_PER_BLOCK);
		printf("size_list :%d , N :%d\n",size_list,N);
		for(int i = 0; i < N ; i++)
		{
			int id = N_THREADS_PER_BLOCK * i + tid;
			if( id <= size_list)
			{
				neb[id] = g_col_index[id + start];
			}
			__syncthreads();
			if( id <= size_list)
			{
				int total = 0;
				total = Intersection(bid, neb[tid], g_col_index, g_row_ptr,neb,size_list);
				//printf("\nedge(%d , %d) : %d",bid, neb[tid], total);
				atomicAdd(&g_sum[0],total);
			}
		}
	}*/
}
int main(int argc, char *argv[])
{
	cudaEvent_t start3,stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	int *col_index;   //CPU MEMORY ALLOCATION
	col_index = (int *)malloc(sizeof(int)*NUM_VERTICES);
	int *g_col_index;  //GPU MEMORY ALOOCATION
	cudaMalloc(&g_col_index,sizeof(int)*NUM_VERTICES);

	int *row_ptr;  //CPU MEMORY ALLOCATION
	row_ptr = (int *)malloc(sizeof(int)*NUM_EDGES);
	int *g_row_ptr;   // GPU MEMORY ALLOCATION
	cudaMalloc(&g_row_ptr,sizeof(int)*NUM_EDGES);

	int edge=0,data=0; //vertex=10670, data allocation from file..

	char *argument2 = argv[2]; //take argument from terminal and initilize
	int vertex=atoi(argument2); //initilize variable

	int *g_sum;
	int *sum;
	sum= (int *)malloc(sizeof(int)*1);
	cudaMalloc((void**)&g_sum,sizeof(int)*1);

	//int nblocks = ceil((float)vertex / BLOCKSIZE);

	//**********file operations***************
	FILE *file;
	file = fopen(argv[1],"r");

	//******************Data From File*******************
	if(file == NULL)
	{
		printf("file not opened\n");
		exit(0);
	}
	else
	{
		fscanf(file , "%d", &edge);
		//printf("\nRow_ptr :");
		for(int i=0; i<vertex+1; i++)
		{
			fscanf(file, "%d", &data);
			row_ptr[i]=data;
			//printf(" %d",data);
		}
		//printf("\nCol_index :");
		for(int j=0; j<edge; j++)
		{
			fscanf(file,"%d", &data);
			col_index[j]=data;
			//printf(" %d",data);
		}
	}
	//**** SEND DATA CPU TO GPU *********************
	cudaMemcpy(g_col_index,col_index,sizeof(int)*NUM_VERTICES,cudaMemcpyHostToDevice);
	cudaMemcpy(g_row_ptr,row_ptr,sizeof(int)*NUM_EDGES,cudaMemcpyHostToDevice);

	//****************KERNEL CALLED *****************
	cudaEventRecord(start3);
	Find_Triangle<<<vertex,N_THREADS_PER_BLOCK>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
	cudaEventRecord(stop3);
	cudaDeviceSynchronize();
	cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
	int Triangle = sum[0];

	cudaEventSynchronize(stop3);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start3, stop3);
	printf("\nSearch : %.4f sec ",milliseconds/1000);
	printf("\tVertex : %d\tEdge : %d\tTriangle : %d ",vertex,edge*2,Triangle);


	//********** FREE THE MEMORY BLOCKS *****************
	free(col_index);
	free(row_ptr);
	cudaFree(g_col_index);
	cudaFree(g_row_ptr);
	printf("\n");
	return 0;
}
