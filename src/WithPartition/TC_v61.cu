#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<math.h>

#define NUM_VERTICES 99999999
#define NUM_EDGES 99999999
#define BLOCKSIZE 1024

__device__ int intersection(int src, int dst, int *d_col_index, int *d_row_ptr, int *d_vertex_arr ,int v_pos,int id ,int total_v_in_partitions)
{
	//******initilized Variables *****************
	int total = 0 ;
	int index = 0;
	int flag = 0;
	int low = id , high = total_v_in_partitions,  mid;
	while(high - low > 1)
	{
		mid = ( high+low )/2;
		if ( d_vertex_arr[mid] < dst ){ low = mid; }
		else if ( d_vertex_arr[mid] > dst ){ high = mid; }
		else
		{
			index = mid;
			flag++;
			break;
		}
	}
	if (flag == 0)
	{
		for(int i=total_v_in_partitions; i<v_pos; i++)
		{
			if( d_vertex_arr[i] == dst){index = i ;}
		}
	}

	int list1_start = d_row_ptr[id];
	int list1_end = d_row_ptr[id+1];
	int list2_start = d_row_ptr[index];
	int list2_end = d_row_ptr[index+1];
	int sizelist2 = list2_end - list2_start;
	//printf(" sizelist2 : %d  ",sizelist2);
	int step = (int)floor(sqrtf(sizelist2));

	while (list1_start < list1_end && list2_start < list2_end)
	{
		if (d_col_index[list1_start] < d_col_index[list2_start]) list1_start++ ;
		else if (d_col_index[list2_start] < d_col_index [list1_start]) list2_start++ ;
		else if (d_col_index[list1_start] == d_col_index[list2_start])
		{
			total++;
			list1_start++;
			list2_start++;
		}
	}
	return total; //return total triangles found by each thread...
}

__global__ void Find_Triangle(int *d_col_index, int *d_row_ptr, int *d_vertex_arr,int total_v_in_partitions, int v_pos, int rp_pos, int ci_pos, int *d_sum )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	if (id < total_v_in_partitions) // only number of vertex thread executed ...
	{
		for (int i = d_row_ptr[id] ; i < d_row_ptr[id+1] ; i++)
		{
			int total = 0;
			total = intersection(d_vertex_arr[id], d_col_index[i], d_col_index, d_row_ptr, d_vertex_arr ,v_pos ,id ,total_v_in_partitions);
			//printf("\n edge(%d , %d) : %d",d_vertex_arr[id], d_col_index[i],total );
			atomicAdd(&d_sum[0],total);
		}
	}
}

int main(int argc, char *argv[])
{
	//------initilization of variables------------//
	int Total_Triangle = 0;

	float total_kernel_time = 0.0 ;
	float total_time = 0.0;

	int v_pos, rp_pos, ci_pos, total_v_in_partitions;

	char *argument2 = argv[2]; //take argument from terminal and initilize
	int vertex=atoi(argument2);

	char *argument3 = argv[3]; //take argument from terminal and initilize
	int edge=atoi(argument3);

	char *argument4 = argv[4];
	int no_partitions = atoi(argument4);

	int *new_col_index;
	cudaMallocHost(&new_col_index,sizeof(int)*NUM_EDGES);

	int *new_row_ptr;
	cudaMallocHost(&new_row_ptr,sizeof(int)*NUM_VERTICES);

	int *vertex_arr;
	cudaMallocHost(&vertex_arr,sizeof(int)*NUM_VERTICES);

	//--------------------Load DATA In Memory---------------------//

	FILE *file;
	file = fopen(argv[1],"r");

	if (file == NULL)
	{
		printf("\nFile Not Operned.........");
		exit(0);
	}
	else
	{
		for (int i = 0; i < no_partitions; i++)
		{
			int data = 0;
			fscanf(file, "%d" , &v_pos);
			fscanf(file, "%d" , &rp_pos);
			fscanf(file, "%d" , &ci_pos);
			fscanf(file, "%d" , &total_v_in_partitions);

			for (int j = 0 ; j < v_pos ; j++)
			{
				fscanf(file, "%d", &data);
				vertex_arr[j]=data;
			}
			for (int j = 0; j < rp_pos; j++)
			{
				fscanf(file, "%d", &data);
				new_row_ptr[j]=data;
			}
			for (int j = 0; j < ci_pos; j++)
			{
				fscanf(file, "%d", &data);
				new_col_index[j]=data;
			}
			//--------------------Launch the kernel-------------------//
			int *d_col_index;  //GPU MEMORY ALOOCATION
			cudaMalloc(&d_col_index,sizeof(int)*ci_pos);

			int *d_vertex_arr;  //GPU MEMORY ALOOCATION
			cudaMalloc(&d_vertex_arr,sizeof(int)*v_pos);

			int *d_row_ptr;   // GPU MEMORY ALLOCATION
			cudaMalloc(&d_row_ptr,sizeof(int)*rp_pos);

			//cudaDeviceSynchronize();
			int *d_sum;
			int *sum;
			sum= (int *)malloc(sizeof(int)*1);
			cudaMalloc((void**)&d_sum,sizeof(int)*1);
			//float total_kernel_time = 0.0 ;

			int nblocks = ceil((float)total_v_in_partitions / BLOCKSIZE);

			//start = clock();
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);

			//--------copy data from host to device --------------//
			cudaMemcpy(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice);
			cudaMemcpy(d_row_ptr,new_row_ptr,sizeof(int)*rp_pos,cudaMemcpyHostToDevice);
			cudaMemcpy(d_vertex_arr,vertex_arr,sizeof(int)*v_pos,cudaMemcpyHostToDevice);

			//---------------------------kernel callled------------------//

			cudaEvent_t startG, stopG;
			cudaEventCreate(&startG);
			cudaEventCreate(&stopG);

			cudaEventRecord(startG);
			Find_Triangle<<<nblocks,BLOCKSIZE>>>(d_col_index,d_row_ptr,d_vertex_arr,total_v_in_partitions,v_pos,rp_pos,ci_pos,d_sum);
			cudaEventRecord(stopG);

			// cudaEventSynchronize(stop);
			//float millisecondsG = 0;
			//cudaEventElapsedTime(&millisecondsG, startG, stopG);
			//printf("    %.4f sec",millisecondsG/1000);
			//total_kernel_time = total_kernel_time + millisecondsG/1000;

			cudaMemcpy(sum,d_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);

			int Triangle = sum[0];
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

			//printf("\t%d\n" , Triangle);
			Total_Triangle = Total_Triangle + Triangle ;


			//free(new_row_ptr);
			//free(new_col_index);
			//free(vertex_arr);
			//cudaFree(d_row_ptr);
			//cudaFree(d_col_index);
			//cudaFree(d_vertex_arr);

		}
		printf("\nTotal Triangle : %d ",Total_Triangle );
		printf("\t Total Kernel Time : %.4f sec",total_kernel_time);
		printf("\t Total Time : %f \n\n",total_time);
	}
	return 0;
}
