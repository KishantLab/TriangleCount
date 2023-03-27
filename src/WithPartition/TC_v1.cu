#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

#define NUM_VERTICES 999999999
#define NUM_EDGES 999999999
#define BLOCKSIZE 1024


//-------------------intersection function ----------------------------------
__device__ int intersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
  //******initilized Variables *****************
  int total = 0 ;
  int pointer1_start = g_row_ptr[src];
  int pointer1_end = g_row_ptr[src+1];
  int pointer2_start = g_row_ptr[dst];
  int pointer2_end = g_row_ptr[dst+1];



  while (pointer1_start < pointer1_end && pointer2_start < pointer2_end) 
  {
    if( src > g_col_index[pointer1_start] && dst > g_col_index[pointer2_start])
    {
      if (g_col_index[pointer1_start] < g_col_index[pointer2_start]) pointer1_start++ ;
      else if (g_col_index[pointer2_start] < g_col_index [pointer1_start]) pointer2_start++ ;
      else if (g_col_index[pointer1_start] == g_col_index[pointer2_start])
      {
        total++;
        pointer1_start++;
        pointer2_start++;
      } 
    }
    else 
    {
      pointer1_start++;
      pointer2_start++;
    } 
  }
/*
  if( src > g_col_index[pointer1_start] && dst > g_col_index[pointer2_start])
  {
  int size_list1 = pointer1_end - pointer1_start;
  int size_list2 = pointer2_end - pointer2_start;

  if (size_list1 < size_list2)
  {
         for (int i=pointer1_start; i<pointer1_end; i++)
          //while (pointer1_start <= pointer1_end )
          {
                  int low = pointer2_start;
                  int high = pointer2_end;
                  int mid = 0 ;
                  while (high-low > 1)
                  {
                          mid = (high + low)/2;
                          if ( g_col_index[mid] < g_col_index[i] ){ low = mid+1; }
                          else if ( g_col_index[mid] > g_col_index[i] ){ high = mid; }
                          else
                          {
                                total++;
                                break;
                          }
                  }
                 //pointer1_start++;
          }
  }
  else
  {       for (int i=pointer2_start; i<pointer2_end; i++)
          //while (pointer2_start <= pointer2_end)
          {
                  int low = pointer1_start;
                  int high = pointer1_end;
                  int mid = 0;
                  while (high - low > 1)
                  {
                          mid = (high + low)/2;
                          if ( g_col_index[mid] < g_col_index[i] ){ low = mid+1; }
                          else if ( g_col_index[mid] > g_col_index[i] ){ high = mid; }
                          else
                          {
                                total++;
                                break;
                          }
                  }
                  //pointer2_start++;
         }
  }
  }*/
  return total; //return total triangles found by each thread...
}

//--------------------------main kernel -------------------------------

__global__ void Find_Triangle(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id 
  //__syncthreads();  // thread barrier

  if (id <= vertex) // only number of vertex thread executed ...
  {
    for (int i = g_row_ptr[id] ; i < g_row_ptr[id+1] ; i++)
    {
      int total = 0;
     //******CALLED INTERSECTION FUNCTION ************
     
      if (id < g_col_index[i])
      {
      total = intersection(id, g_col_index[i], g_col_index, g_row_ptr );
      //g_sum = g_sum + total;
      atomicAdd(&g_sum[0],total);
      }
    }
  }
}

int main(int argc, char *argv[])
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

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
  char *argument3 = argv[3]; // //take argument from terminal and initilize
  int triangle=atoi(argument3);  //initilize variable
  
  int *g_sum;  
  int *sum;
  sum= (int *)malloc(sizeof(int)*1);
  cudaMalloc((void**)&g_sum,sizeof(int)*1);

  int nblocks = ceil((float)vertex / BLOCKSIZE);
  //printf("Total Number of Blocks : %d and threads : %d Launched..\n",nblocks,BLOCKSIZE*nblocks);

  //FILE *file;
  //file = fopen("/content/drive/MyDrive/Kishan_Tamboli/Automation/oregon1_010331_adj.txt","r");

  //**********file operations***************
  FILE *file;
  file = fopen(argv[1],"r");
  char *file_name = argv[4];
  printf("%s, ",file_name);
  printf("%d, ",vertex);

  //******************Data From File*******************
  if(file == NULL)
  {
    printf("file not opened\n");
    exit(0);
  }
  else
  {
    fscanf(file , "%d", &edge);
    for(int i=0; i<=vertex+1; i++)
    {
      fscanf(file, "%d", &data);
      row_ptr[i]=data;
    }
    for(int j=0; j<edge; j++)
    {
      //if(j==edge){col_index[j]=-1;}
      //else
      //{
        fscanf(file,"%d", &data);
        col_index[j]=data;
      //}
    }
  }
  //**** SEND DATA CPU TO GPU *********************
  cudaMemcpy(g_col_index,col_index,sizeof(int)*NUM_VERTICES,cudaMemcpyHostToDevice);
  cudaMemcpy(g_row_ptr,row_ptr,sizeof(int)*NUM_EDGES,cudaMemcpyHostToDevice);

  //****************KERNEL CALLED *****************

  cudaEventRecord(start);
  Find_Triangle<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);

  
  //***********TAKE DATA FROM GPU TO CPU************
  cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
  int Triangle = sum[0];
  //printf(" Total Triangles : %d\n ", Triangle);
  
  //*********PRINTING THE DATA *******************
  printf("%d, ",edge);
  printf("%d, ",triangle);
  printf("%d, ",Triangle);
  printf("%d, ",triangle-Triangle);
  
  //*******CHECKING DATA CORRECT OR NOT ****************
  if (triangle-Triangle == 0)
  { printf("Correct, "); }
  else
  { printf("Not Correct, "); }

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("%.4f sec \n",milliseconds/1000);
  
  //********** FREE THE MEMORY BLOCKS *****************
  free(col_index);
  free(row_ptr);
  cudaFree(g_col_index);
  cudaFree(g_row_ptr);

  return 0;
}
