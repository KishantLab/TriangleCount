#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<math.h>

#define NUM_VERTICES 99999999
#define NUM_EDGES 99999999
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
__global__ void Find_Triangle(int *d_col_index, int *d_row_ptr, int *d_vertex_arr,int total_v_in_partitions, int v_pos, int rp_pos, int ci_pos, int *d_sum )
{
	//int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	__shared__ int start;
	__shared__ int end;
	__shared__ int neb[SHARED_MEM];

	if (tid == 0)
	{
		start = d_row_ptr[bid];
		end = d_row_ptr[bid+1]-1;
	}
	__syncthreads();
	int size_list1 = end - start;
	unsigned long long int triangle = 0;

	if(size_list1 < N_THREADS_PER_BLOCK)
	{
		if(tid <= size_list1)
		{
			neb[tid] = d_col_index[tid+start];
		}
		__syncthreads();
		for( int i = 0; i <= size_list1; i++)
		{
			int index = 0;
			int flag = 0;
			int low = 0 , high = total_v_in_partitions,  mid;
			int dst = neb[i];
			while(low <= high)
			{
				mid = ( high+low )/2;
				if ( d_vertex_arr[mid] < dst ){ low = mid+1; }
				else if ( d_vertex_arr[mid] > dst ){ high = mid-1; }
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
			int start2 = d_row_ptr[index];
			int end2 = d_row_ptr[index+1]-1;
			int size_list2 = end2 - start2;
			int M = ceil((float)(size_list2 +1)/N_THREADS_PER_BLOCK);
			for( int k = 0; k < M; k++)
			{
				int id = N_THREADS_PER_BLOCK * k + tid;
				if(id <= size_list2)
				{
					int result = 0;
					result = Search(d_col_index[id+start2],neb,size_list1);
					//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d skey:%d, neb[0]:%d ,neb[%d]:%d",d_vertex_arr[bid], neb[i], result,tid,size_list1+1,size_list2+1,start2,end2,d_col_index[id+start2],neb[0],size_list1,neb[size_list1]);
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
					int index = 0;
					int flag = 0;
					int low = 0 , high = total_v_in_partitions,  mid;
					int dst = neb[i];
					while(low <= high)
					{
						mid = ( high+low )/2;
						if ( d_vertex_arr[mid] < dst ){ low = mid+1; }
						else if ( d_vertex_arr[mid] > dst ){ high = mid-1; }
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
							if( d_vertex_arr[i] == dst){index = i;}
						}
					}
					int start2 = d_row_ptr[index];
					int end2 = d_row_ptr[index+1]-1;
					int size_list2 = end2 - start2;
					int M = ceil((float)(size_list2 +1)/N_THREADS_PER_BLOCK);
					for( int k = 0; k < M; k++)
					{
						int tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2)
						{
							int result = 0;
							result = Search(d_col_index[tempid+start2],neb,size);
							//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d, remining_size:%d, size:%d, neb[0]:%d, neb[%d]:%d if ",d_vertex_arr[bid], d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,remining_size,size,neb[0],size,neb[size]);
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

				if(id <= size_list1)
				{
					neb[tid] = d_col_index[id+start];
					//printf(" neb : %d", neb[tid]);
				}
				__syncthreads();
				for( int j = start; j <= end; j++)
				{
					int index = 0;
					int flag = 0;
					int low = 0 , high = total_v_in_partitions,  mid;
					int dst = neb[i];
					while(low <= high)
					{
						mid = ( high+low )/2;
						if ( d_vertex_arr[mid] < dst ){ low = mid+1; }
						else if ( d_vertex_arr[mid] > dst ){ high = mid-1; }
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
					int start2 = d_row_ptr[index];
					int end2 = d_row_ptr[index+1]-1;
					int size_list2 = end2 - start2;
					int M = ceil((float)(size_list2 +1)/ N_THREADS_PER_BLOCK);
					for (int k = 0; k < M; k++)
					{
						int tempid = N_THREADS_PER_BLOCK * k + tid;
						if(tempid <= size_list2)
						{
							int result = 0;
							result = Search(d_col_index[tempid+start2],neb,remining_size);
							//printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d neb[0]:%d, neb[%d]:%d, else",d_vertex_arr[bid], d_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,d_col_index[tempid+start2],N,i,neb[0],remining_size,neb[remining_size]);
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
	atomicAdd(&d_sum[0],triangle);
}
/*
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
*/
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

			//---------------------------kernel callled------------------//

			cudaEvent_t startG, stopG;
			cudaEventCreate(&startG);
			cudaEventCreate(&stopG);

			cudaEventRecord(startG);
			Find_Triangle<<<total_v_in_partitions,N_THREADS_PER_BLOCK>>>(d_col_index,d_row_ptr,d_vertex_arr,total_v_in_partitions,v_pos,rp_pos,ci_pos,d_sum);
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

			printf("\t%d\n" , Triangle);
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
