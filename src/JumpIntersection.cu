#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<time.h>

#define NUM_VERTICES 99999999
#define NUM_EDGES 99999999
#define BLOCKSIZE 1024

//-------------------intersection function ----------------------------------
__device__ int JumpIntersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
	int total = 0 ;
	int list1_start = g_row_ptr[src];
	int list1_end = g_row_ptr[src+1];
	int list2_start = g_row_ptr[dst];
	int list2_end = g_row_ptr[dst+1];
	int sizelist2 = list2_end - list2_start;
	int step = (int)floor(sqrtf(sizelist2));
	for(int x =  list1_start; x < list1_end; x++)
	{
		if(g_col_index[list2_start] > g_col_index[x])
		{
			continue;
		}
		else
		{
			int start = list2_start;
			int end = step + list2_start;
			while( g_col_index[end-1] < g_col_index[x] && start < list2_end)
			{
				start = end;
				end = end +step;
				if( end > list2_end)
				{
					end = list2_end;
					break;
				}
			}
			
			int lo = start , hi = end-1, mi;
			while(lo <= hi)
			{
				mi = (hi+lo)/2;
				if(g_col_index[mi] < g_col_index[x]){lo=mi+1;}
				else if(g_col_index[mi] > g_col_index[x]){hi = mi-1;}	
				else{
					total++;
					break;
				}
			}
			/*

			for( int y = start; y < end; y++)
			{
				if(g_col_index[y] == g_col_index[x])
				{
					total++;
				}
			}
			*/
			
		}
	}
	return total; //return total triangles found by each thread...
}

__global__ void Find_Triangle_Jump(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	if (id < vertex) // only number of vertex thread executed ...
	{
		for (int i = g_row_ptr[id] ; i < g_row_ptr[id+1] ; i++)
		{
			int total = 0;
			total = JumpIntersection(id, g_col_index[i], g_col_index, g_row_ptr );
			//printf("\nedge(%d , %d) : %d",id, g_col_index[i], total);
			atomicAdd(&g_sum[0],total);
		}
	}
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
	Find_Triangle_Jump<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
	cudaEventRecord(stop3);
	cudaDeviceSynchronize();
	cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
	int Triangle_Jump = sum[0];
	
	cudaEventSynchronize(stop3);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start3, stop3);
	printf("\nSearch : %.4f sec",milliseconds/1000);
	printf("\tVertex : %d\tEdge : %d\tTriangle : %d ",vertex,edge,Triangle_Jump);


	//********** FREE THE MEMORY BLOCKS *****************
	//free(col_index);
	//free(row_ptr);
	//cudaFree(g_col_index);
	//cudaFree(g_row_ptr);
	printf("\n");
	return 0;
}
