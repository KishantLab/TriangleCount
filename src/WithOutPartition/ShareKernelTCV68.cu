#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<time.h>

#define NUM_VERTICES 999999999
#define NUM_EDGES 999999999
#define N_THREADS_PER_BLOCK 256
#define SHARED_MEM 256

//-------------------intersection function ----------------------------------
__device__ int Search (int skey , int *neb, int sizelist, int lo, int hi)
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
__global__ void Find_Triangle(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
{
	//int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	__shared__ int start;
	__shared__ int end;
	__shared__ int neb[SHARED_MEM];

	//int start = g_row_ptr[bid];
	//int end = g_row_ptr[bid+1]-1;
	//int index = reordered_array[bid];
	if(tid ==0)
	{
		start = g_row_ptr[bid];
		end = g_row_ptr[bid+1]-1;
	}
	__syncthreads();
	int size_list1 = end - start;
	int triangle = 0;
	//if(size_list1 ==0 ) return;
	if(size_list1 < N_THREADS_PER_BLOCK)
	{
		if(tid <= size_list1)
		{
			neb[tid] = g_col_index[tid+start];
		}
		__syncthreads();
		for( int i = 0; i <= size_list1; i++)
		{
			int start2 = g_row_ptr[neb[i]];
			int end2 = g_row_ptr[neb[i]+1]-1;
			int size_list2 = end2 - start2;
			int M = ceil((float)(size_list2 +1)/N_THREADS_PER_BLOCK);
			for( int k = 0; k < M; k++)
			{
				int id = N_THREADS_PER_BLOCK * k + tid;
				int lo = 0;
				int hi = size_list1;
				if(id <= size_list2) 
				{
					int result = 0;
					result = Search(g_col_index[id+start2],neb,size_list1,lo,hi);
					//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d skey:%d, neb[0]:%d ,neb[%d]:%d",bid, neb[i], result,tid,size_list1+1,size_list2+1,start2,end2,g_col_index[id+start2],neb[0],size_list1,neb[size_list1]);
					//atomicAdd(&g_sum[0],result);
					//printf("\nedge(%d , %d) src : %d dst :%d ", bid,neb[i],size_list1+1,size_list2+1);
					if(result > 0 && result != index)
						triangle = triangle +1;
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
			int index = 0;
			if( remining_size > size)
			{
				if(id <= size_list1)
				{
					neb[tid] = g_col_index[id+start];
					//printf(" neb : %d", neb[tid]);
				}
				__syncthreads();
				for( int j = start; j <= end; j++)
				{
					int start2 = g_row_ptr[g_col_index[j]];
					int end2 = g_row_ptr[g_col_index[j]+1]-1;
					int size_list2 = end2 - start2;
					int hi = size;
					int lo = index;
					int M = ceil((float)(size_list2 +1)/N_THREADS_PER_BLOCK);
					for( int k = 0; k < M; k++)
					{
						int tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2) 
						{
							int result = 0;
							result = Search(g_col_index[tempid+start2],neb,size,lo,hi);
							//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d, remining_size:%d, size:%d, neb[0]:%d, neb[%d]:%d if ",bid, g_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,g_col_index[tempid+start2],N,i,remining_size,size,neb[0],size,neb[size]);
							//atomicAdd(&g_sum[0],result);
							//printf("\nedge(%d , %d) src : %d dst :%d ", bid,g_col_index[j],size_list1+1,size_list2+1);
							if(result > 0 && result != index)
								triangle = triangle+1;
						}
					}
				}
				__syncthreads();
				remining_size = remining_size-(size+1);
			}
			else
			{

				if(id <= size_list1)
				{
					neb[tid] = g_col_index[id+start];
					//printf(" neb : %d", neb[tid]);
				}
				__syncthreads();
				for( int j = start; j <= end; j++)
				{
					int start2 = g_row_ptr[g_col_index[j]];
					int end2 = g_row_ptr[g_col_index[j]+1]-1;
					int size_list2 = end2 - start2;
					int M = ceil((float)(size_list2 +1)/ N_THREADS_PER_BLOCK);
					for (int k = 0; k < M; k++)
					{
						int tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2) 
						{
							int result = 0;
							result = Search(g_col_index[tempid+start2],neb,remining_size);
							//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d neb[0]:%d, neb[%d]:%d, else",bid, g_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,g_col_index[tempid+start2],N,i,neb[0],remining_size,neb[remining_size]);
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
	atomicAdd(&g_sum[0],triangle);
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
		for(int i=0; i<=vertex+1; i++)
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
	//printf("\nSearch : %.4f sec ",milliseconds/1000);
	printf("\nSearch : %.6f sec Vertex : %d Edge : %d Triangle : %d ",milliseconds/1000,vertex,edge*2,Triangle);


	//********** FREE THE MEMORY BLOCKS *****************
	free(col_index);
	free(row_ptr);
	cudaFree(g_col_index);
	cudaFree(g_row_ptr);
	printf("\n");
	return 0;
}
