#include<stdio.h>
#include<stdlib.h>

#define NUM_VERTICES 9999999999
#define NUM_EDGES 9999999999
#define BLOCKSIZE 1024

__device__ int intersection(int src, int dst, int *d_col_index, int *d_row_ptr, int *d_vertex_arr ,int v_pos,int id ,int total_v_in_partitions, int NP)
{
  //******initilized Variables *****************
  printf("\npart : %d, src : %d , dst : %d ",NP,src,dst);
  int total = 0 ;
  int pointer1_start = d_row_ptr[id];
  int pointer1_end = d_row_ptr[id+1];

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

  int pointer2_start = d_row_ptr[index];
  int pointer2_end = d_row_ptr[index+1];

  while (pointer1_start < pointer1_end && pointer2_start < pointer2_end)
  {
    if (d_col_index[pointer1_start] < d_col_index[pointer2_start]) pointer1_start++ ;
    else if (d_col_index[pointer2_start] < d_col_index [pointer1_start]) pointer2_start++ ;
    else if (d_col_index[pointer1_start] == d_col_index[pointer2_start])
    {
      total++;
      pointer1_start++;
      pointer2_start++;
    }
  }

  return total; //return total triangles found by each thread...
}

__global__ void Find_Triangle(int *d_col_index, int *d_row_ptr, int *d_vertex_arr,int total_v_in_partitions, int v_pos, int rp_pos, int ci_pos, int *d_sum, int NP )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id

  if (id < total_v_in_partitions) // only number of vertex thread executed ...
  {
    for (int i = d_row_ptr[id] ; i < d_row_ptr[id+1] ; i++)
    {
      int total = 0;

      //******CALLED INTERSECTION FUNCTION ************
      total = intersection(d_vertex_arr[id], d_col_index[i], d_col_index, d_row_ptr, d_vertex_arr ,v_pos ,id ,total_v_in_partitions,NP);
      atomicAdd(&d_sum[0],total);

    }

  //printf("Total Triangle p: %d",d_sum[0]);
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
  new_col_index= (int *) malloc(sizeof(int)*edge);

  int *new_row_ptr;
  new_row_ptr = (int *) malloc(sizeof(int)*vertex);

  int *vertex_arr;
  vertex_arr = (int *) malloc(sizeof(int)*vertex);

  int *v_pos_arr;
  v_pos_arr = (int *) malloc(sizeof(int)*vertex);

  int *rp_pos_arr;
  cudaMallocHost(&rp_pos_arr,sizeof(int)*no_partitions+1);

  int *ci_pos_arr;
  cudaMallocHost(&ci_pos_arr,sizeof(int)*no_partitions+1);

  int *total_v_in_partitions_arr;
  total_v_in_partitions_arr = (int *) malloc(sizeof(int)*no_partitions+1);

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
    v_pos_arr[0] = 0;
    ci_pos_arr[0] = 0;
    rp_pos_arr[0] = 0;
    total_v_in_partitions_arr[0] = 0;

    for (int i = 0; i < no_partitions; i++)
    {
      int data = 0;
      fscanf(file, "%d" , &v_pos);
      v_pos_arr[i+1] = v_pos;
      printf("v_pos_arr[%d]:%d  ",i+1,v_pos);

      fscanf(file, "%d" , &rp_pos);
      rp_pos_arr[i+1] = rp_pos;
      printf("rp_pos_arr[%d]: %d  ",i+1,rp_pos);

      fscanf(file, "%d" , &ci_pos);
      ci_pos_arr[i+1] = ci_pos;
      printf("ci_pos_arr[%d]: %d  ",i+1,ci_pos);

      fscanf(file, "%d" , &total_v_in_partitions);
      total_v_in_partitions_arr[i+1] = total_v_in_partitions;
      printf("total_v_in_partitions_arr[%d]: %d  ",i+1,total_v_in_partitions);

      for (int j = v_pos_arr[i] ; j < v_pos_arr[i+1]; j++)
      {
        fscanf(file, "%d", &data);
        vertex_arr[j]=data;
      }
      printf("\nVertex Array Loaded....");
      for (int j = rp_pos_arr[i] ; j < rp_pos_arr[i+1]; j++)
      {
        fscanf(file, "%d", &data);
        new_row_ptr[j]=data;
      }
      printf("\nRow Pointer Array Loaded.....");

      for (int j = ci_pos_arr[i] ; j< ci_pos_arr[i+1]; j++)
      {
        fscanf(file, "%d", &data);
        new_col_index[j]=data;
      }
      printf("\nCol Index Array Loaded.......");
    }
    printf("\n");
  }


//==============================CREATE STREAMS================================//
  cudaStream_t stream[no_partitions];
  printf("\nGPU Stream Created.............");

  int *d_col_index;  //GPU MEMORY ALOOCATION
  cudaMalloc(&d_col_index,sizeof(int)*ci_pos);

  int *d_vertex_arr;  //GPU MEMORY ALOOCATION
  cudaMalloc(&d_vertex_arr,sizeof(int)*v_pos);

  int *d_row_ptr;   // GPU MEMORY ALLOCATION
  cudaMalloc(&d_row_ptr,sizeof(int)*rp_pos);

  printf("\nGPU Arrays Created...............");
  int *d_sum;
  int *sum;
  cudaMallocHost(&sum,sizeof(int)*1);
  cudaMalloc((void**)&d_sum,sizeof(int)*1);

  int nblocks = ceil((float)total_v_in_partitions / BLOCKSIZE);

  printf("\nBlocks : %d ", nblocks);
  cudaEvent_t start, stop, startG, stopG;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  printf("\nEvent Initilize......");
  cudaEventCreate(&startG);
  printf("\nFirst Event Created.......");
  cudaEventCreate(&stopG);
  printf("\nSecend Event created..........");

  for (int i=0; i<no_partitions; i++){cudaStreamCreate(&stream[i]);}  //create stream for Device

  //--------copy data from host to device --------------//
  cudaEventRecord(start);
  printf("\nStart Copy Data From Host To Device ......");
  for (int j = 0; j < no_partitions; j++)
  {
    printf("\nCopy start For P:%d.....",j);
    int offsetv = v_pos_arr[j+1];
	printf("\nOffset v_pos : %d", offsetv);
    int offsetr = rp_pos_arr[j+1];
    int offsetc = ci_pos_arr[j+1];

    cudaMemcpyAsync(d_row_ptr,new_row_ptr,sizeof(int)*offsetr,cudaMemcpyHostToDevice,stream[j]);
    cudaMemcpyAsync(d_col_index,new_col_index,sizeof(int)*offsetc,cudaMemcpyHostToDevice,stream[j]);
    cudaMemcpyAsync(d_vertex_arr,vertex_arr+v_pos_arr[j],sizeof(int)*offsetv,cudaMemcpyHostToDevice,stream[j]);
    //cudaStreamSynchronize(stream[j]);

    printf("\nCopy Completed for P:%d....",j);
  }
  printf("\nKernel Called...................");
  cudaEventRecord(startG);
  printf("\nStart Kernel..........");
  for (int k = 0 ; k < no_partitions; k++)
  {
    //cudaStreamSynchronize(stream[k]);
    printf("\nKernel Called for P:%d....",k);
    Find_Triangle<<<nblocks,BLOCKSIZE,0,stream[k]>>>(d_col_index,d_row_ptr,d_vertex_arr,total_v_in_partitions_arr[k+1],v_pos_arr[k+1],rp_pos_arr[k+1],ci_pos_arr[k+1],d_sum,k);
  }
  cudaEventRecord(stopG);
  printf("\nKernel Execution Done .....");

  for( int j=0 ; j < no_partitions; j++)
  {
    cudaMemcpyAsync(sum,d_sum,sizeof(int)*1,cudaMemcpyDeviceToHost,stream[j]);
   	//int Triangle = sum[0];
   	//printf("\t%d\n" , Triangle);
  	//total_Triangle = Total_Triangle + Triangle ;
  }

  for( int j=0 ; j < no_partitions; j++)
  {
	//cuStreamSynchonize(stream[j]);
	//cudaError_t cudaStreamSynchronize();
	int Triangle = sum[0];

   	//printf("\t%d\n" , Triangle);
  	Total_Triangle = Total_Triangle + Triangle ;

  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float millisecondsG = 0;
  cudaEventElapsedTime(&millisecondsG, startG, stopG);
  printf("\nGPU Time : %.4f sec",millisecondsG/1000);
  total_kernel_time = total_kernel_time + millisecondsG/1000;

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\nTotal Time :%.4f sec",milliseconds/1000);
  total_time = total_time + milliseconds/1000;

  printf("\nTotal Triangle : %d ",Total_Triangle);
  printf("\n");

  return 0;
}
