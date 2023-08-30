#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<time.h>

#define N_THREADS_PER_BLOCK 256
//#define SHARED_MEM 1024

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

//-------------------intersection function ----------------------------------
__device__ int Search (unsigned long long int skey , unsigned long long int *neb, unsigned long long int end, unsigned long long int start)
{
	unsigned long long int total = 0;
	if(skey < neb[start] || skey > neb[end])
	{
		return 0;
	}
	else if(skey == neb[start] || skey == neb[end])
	{
		return 1;
	}
	else
	{
		unsigned long long int lo = start+1;
		unsigned long long int hi = end-1;
		unsigned long long int mid=0;
		while( lo <= hi)
		{
			mid = (hi+lo)/2;
			//printf("\nskey :%llu , mid : %llu ",skey,neb[mid]);
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
__global__ void Find_Triangle(unsigned long long int *g_col_index, unsigned long long int *g_row_ptr ,unsigned long long int *g_sum )
{
  __shared__ unsigned long long int s_sum[N_THREADS_PER_BLOCK];
	unsigned long long int bid = blockIdx.x;
	unsigned long long int tid = threadIdx.x;
  unsigned long long int start;
	unsigned long long int end;
	start = g_row_ptr[bid];
	end = g_row_ptr[bid+1]-1;
	//unsigned long long int size_list1 = end - start;
	unsigned long long int triangle = 0;
		for( unsigned long long int i = start; i <= end; i++)
		{
			unsigned long long int start2 = g_row_ptr[g_col_index[i]];
			unsigned long long int end2 = g_row_ptr[g_col_index[i]+1]-1;
			unsigned long long int size_list2 = end2 - start2;
			unsigned long long int M = ceil((float)(size_list2+1)/N_THREADS_PER_BLOCK);
			for( unsigned long long int k = 0; k < M; k++)
			{
				unsigned long long int id = N_THREADS_PER_BLOCK * k + tid;
				if(id <= size_list2)
				{
					unsigned long long int result = 0;
					result = Search(g_col_index[id+start2],g_col_index,end,start);
					//printf("\nedge(%llu , %llu) : %llu , tid : %llu, size_list1 :%llu , size_list2: %llu, start2 :%llu , end2 :%llu skey:%llu, g_col_index[0]:%llu ,g_col_index[%llu]:%llu",bid, g_col_index[i], result,tid,size_list1+1,size_list2+1,start2,end2,g_col_index[id+start2],g_col_index[start],size_list1,g_col_index[end]);
					//atomicAdd(&g_sum[0],result);
					//printf("\nedge(%llu , %llu) src : %llu dst :%llu ", bid,neb[i],size_list1+1,size_list2+1);
					triangle += result;
				}
			}
		}
	//atomicAdd(&g_sum[0],triangle);
  s_sum[tid] = triangle;
    __syncthreads();
    if (tid == 0)
    {
        unsigned long long int block_sum = 0;
        for (int i = 0; i < N_THREADS_PER_BLOCK; i++)
        {
            block_sum += s_sum[i];
        }
        g_sum[bid] = block_sum;
    }
}
int main(int argc, char *argv[])
{
	cudaEvent_t start3,stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	unsigned long long int Edges=0,data=0,Vertex=0, row_ptr_s=0, col_idx_s=0; //vertex=10670, data allocation from file..
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
    fscanf(file , "%llu", &Vertex);
    fscanf(file , "%llu", &Edges);
		fscanf(file , "%llu", &row_ptr_s);
		fscanf(file , "%llu", &col_idx_s);

		unsigned long long int *row_ptr;  //CPU MEMORY ALLOCATION
		row_ptr = (unsigned long long int *)malloc(sizeof(unsigned long long int)*row_ptr_s);
    unsigned long long int *col_index;   //CPU MEMORY ALLOCATION
    col_index = (unsigned long long int *)malloc(sizeof(unsigned long long int)*col_idx_s);

		for(unsigned long long int i=0; i<row_ptr_s; i++)
		{
			fscanf(file, "%llu", &data);
			row_ptr[i]=data;
			//printf(" %llu",data);
		}
		//printf("\nCol_index :");
		for(unsigned long long int j=0; j<col_idx_s; j++)
		{
			fscanf(file,"%llu", &data);
			col_index[j]=data;
			//printf(" %llu",data);
		}

		unsigned long long int *g_row_ptr;   // GPU MEMORY ALLOCATION
		cudaMalloc(&g_row_ptr,sizeof(unsigned long long int)*row_ptr_s);
    unsigned long long int *g_col_index;  //GPU MEMORY ALOOCATION
		cudaMalloc(&g_col_index,sizeof(unsigned long long int)*col_idx_s);
    // int *neb;  //GPU MEMORY ALOOCATION
    // cudaMalloc(&neb,sizeof( int)*N_THREADS_PER_BLOCK);

		//**** SEND DATA CPU TO GPU *********************
    cudaMemcpy(g_row_ptr,row_ptr,sizeof(unsigned long long int)*row_ptr_s,cudaMemcpyHostToDevice);
		cudaMemcpy(g_col_index,col_index,sizeof(unsigned long long int)*col_idx_s,cudaMemcpyHostToDevice);

		unsigned long long int *sum;
		sum = (unsigned long long int *)malloc(sizeof(unsigned long long int)*1);

		unsigned long long int *g_sum;
		cudaMalloc((void**)&g_sum,sizeof(unsigned long long int)*1);

		//****************KERNEL CALLED *****************
		cudaEventRecord(start3);
		Find_Triangle<<<Vertex,N_THREADS_PER_BLOCK>>>(g_col_index,g_row_ptr,g_sum);
		cudaEventRecord(stop3);
		cudaDeviceSynchronize();
		cudaMemcpy(sum,g_sum,sizeof(unsigned long long int)*1,cudaMemcpyDeviceToHost);
		unsigned long long int Triangle = sum[0];

		cudaEventSynchronize(stop3);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start3, stop3);
		//printf("\nSearch : %.4f sec ",milliseconds/1000);
		printf("\nSearch : %.6f sec Vertex : %llu Edge : %llu Triangle : %llu\n",milliseconds/1000,Vertex,col_idx_s,Triangle);

		//********** FREE THE MEMORY BLOCKS *****************
		free(col_index);
		free(row_ptr);
		free(sum);
		cudaFree(g_col_index);
		cudaFree(g_row_ptr);
		cudaFree(g_sum);
	}
	//printf("\n");
	return 0;
}
