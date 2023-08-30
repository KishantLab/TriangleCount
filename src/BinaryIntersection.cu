#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<time.h>

#define NUM_VERTICES 9999999999
#define NUM_EDGES 9999999999
#define BLOCKSIZE 1024

//-------------------intersection function ----------------------------------
__device__ int Intersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
	int total = 0 ;
	int list1_start = g_row_ptr[src];
	int list1_end = g_row_ptr[src+1];
	int list2_start = g_row_ptr[dst];
	int list2_end = g_row_ptr[dst+1];

	for(int x = list1_start; x < list1_end; x++)
	{
		//if(g_col_index[x] < g_col_index[list2_start])
		//{
			//continue;
		//}
		//else
		//{
			int lo = list2_start;
			int hi = list2_end-1;
			int mid=0;
			while( lo <= hi)
			{
				mid = (hi+lo)/2;
				if( g_col_index[mid] < g_col_index[x]){lo=mid+1;}
				else if(g_col_index[mid] > g_col_index[x]){hi=mid-1;}
				else if(g_col_index[mid] == g_col_index[x])
				{
					total++;
					break;
				}

			}
		//}
	}
	return total; //return total triangles found by each thread...
}

__global__ void Find_Triangle(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum, int *g_weight_arr)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	if (id < vertex) // only number of vertex thread executed ...
	{
		for (int i = g_row_ptr[id] ; i < g_row_ptr[id+1] ; i++)
		{
			int total = 0;
			total = Intersection(id, g_col_index[i], g_col_index, g_row_ptr );
			//printf("\nedge(%d , %d) : %d",id, g_col_index[i], total);
			g_weight_arr[i] = total;
			atomicAdd(&g_sum[0],total);
		}
	}
}

int main(int argc, char *argv[])
{
	printf("Hello");
	cudaEvent_t start3,stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	int edge=0,data=0; //vertex=10670, data allocation from file..
	char *argument2 = argv[2]; //take argument from terminal and initilize
	int vertex=atoi(argument2); //initilize variable
	vertex = vertex+2;
	int *g_sum;
	int *sum;
	sum= (int *)malloc(sizeof(int)*1);
	cudaMalloc((void**)&g_sum,sizeof(int)*1);

	int *row_ptr;  //CPU MEMORY ALLOCATION
	row_ptr = (int *)malloc(sizeof(int)*NUM_VERTICES);
	int *g_row_ptr;   // GPU MEMORY ALLOCATION
	cudaMalloc(&g_row_ptr,sizeof(int)*NUM_VERTICES);

	int *col_index;   //CPU MEMORY ALLOCATION
	col_index = (int *)malloc(sizeof(int)*NUM_EDGES);
	int *g_col_index;  //GPU MEMORY ALOOCATION
	cudaMalloc(&g_col_index,sizeof(int)*NUM_EDGES);

	int *weight_arr;
	weight_arr = (int *)malloc(sizeof(int)*NUM_EDGES);
	int *g_weight_arr;
	cudaMalloc(&g_weight_arr,sizeof(int)*NUM_EDGES);

	int nblocks = ceil((float)vertex / BLOCKSIZE);

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
		for(int i=0; i<vertex; i++)
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
	cudaMemcpy(g_col_index,col_index,sizeof(int)*edge,cudaMemcpyHostToDevice);
	cudaMemcpy(g_row_ptr,row_ptr,sizeof(int)*vertex,cudaMemcpyHostToDevice);

	//****************KERNEL CALLED *****************
	cudaEventRecord(start3);
	Find_Triangle<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum,g_weight_arr);
	cudaEventRecord(stop3);
	cudaDeviceSynchronize();
	cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
	cudaMemcpy(weight_arr,g_weight_arr,sizeof(int)*edge,cudaMemcpyDeviceToHost);
	int Triangle = sum[0];

	printf("\n weight_arr: ");
	for(int i=0; i<edge; i++)
	{
		printf(" %d", weight_arr[i]);
	}
	cudaEventSynchronize(stop3);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start3, stop3);
	printf("\nSearch : %.4f sec ",milliseconds/1000);
	printf("\tVertex : %d\tEdge : %d\tTriangle : %d ",vertex,edge,Triangle/6);


	//********** FREE THE MEMORY BLOCKS *****************
	//free(col_index);
	//free(row_ptr);
	//cudaFree(g_col_index);
	//cudaFree(g_row_ptr);
	printf("\n");
	return 0;
}
