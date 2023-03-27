#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

#define NUM_VERTICES 999999999
#define NUM_EDGES 999999999
#define BLOCKSIZE 1024

//int min(int num1, int num2)
//{
//    return (num1 > num2 ) ? num2 : num1;
//}
//-------------------intersection function ----------------------------------
__device__ int TwoPointerIntersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
  //******initilized Variables *****************
  int total = 0 ;
  int pointer1_start = g_row_ptr[src];
  //printf("\npointer1_start_2pointer : %d",pointer1_start);
  int pointer1_end = g_row_ptr[src+1];
  //printf("\nPointer1_end_2pointer : %d ",pointer1_end); 
  int pointer2_start = g_row_ptr[dst];
  //printf("\npointer2_start_2pointer : %d",pointer2_start);
  int pointer2_end = g_row_ptr[dst+1];
  //printf("\npointer2_end_2pointer : %d",pointer2_end);



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

 
  return total; //return total triangles found by each thread...
}
__device__ int BinaryIntersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
  int total = 0 ;
  int list1_start = g_row_ptr[src];
  //printf("\nlist1_start_b : %d ",list1_start);
  int list1_end = g_row_ptr[src+1];
  //printf("\nlist1_end_b : %d",list1_end);
  int list2_start = g_row_ptr[dst];
  //printf("\nlist2_start_b : %d",list2_start);
  int list2_end = g_row_ptr[dst+1];
  //printf("\nlist2_end_b : %d",list2_end);

  //int size_list1 = list1_end - list1_start;
  //int size_list2 = list2_end - list2_start;

  while (list1_start < list1_end && list2_start < list2_end) 
  { 
	  if( src > g_col_index[list1_start] && dst > g_col_index[list2_start])
	  {

		  for (int i=list1_start; i<list1_end; i++)
			  //while (pointer1_start <= pointer1_end )
		  {
			  int low = list2_start;
			  int high = list2_end;
			  int mid = 0 ;
			  while (high-low > 1)
			  {
				  mid = (high + low)/2;
				  if ( g_col_index[mid] < g_col_index[i] ){ low = mid+1; }
				  else if ( g_col_index[mid] > g_col_index[i] ){ high = mid; }
				  else if( g_col_index[mid] == g_col_index[i]) 
				  {
					  total++;
					  list1_start++;
					  list2_start++;
					  break;
					  //continue;
				  }
			  }
			  list1_start++;
			  list2_start++;
			  //pointer1_start++;
		  }
		  list1_start++;
		  list2_start++;
	  }
	  else
	  {
		  list1_start++;
		  list2_start++;
	  }
  }
  return total; //return total triangles found by each thread...
}

__device__ int JumpIntersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
  int total = 0 ;
  int list1_start = g_row_ptr[src];
  int list1_end = g_row_ptr[src+1];
  int list2_start = g_row_ptr[dst];
  int list2_end = g_row_ptr[dst+1];

  int sizelist2 = list2_end - list2_start;
  int step = (int)floor(sqrtf(sizelist2));
  //while (list1_start < list1_end && list2_start < list2_end) 
  //{ 
	  if( src > g_col_index[list1_start] && dst > g_col_index[list2_start])
	  {


		  for (int x=list1_start; x<list1_end; x++)
		  {
			  int prev = 0; 
			  int block = step;

			  while( g_col_index[min(block,sizelist2)-1] < g_col_index[x])
			  {
				  prev = block;
				  block += (int)floor(sqrtf(sizelist2));	       
				  if (prev > sizelist2)
				  {
					  list1_start++;
					  list2_start++;
					  break;
				  }
			  }

			  while(g_col_index[prev] < g_col_index[x])
			  {
				  prev = prev+1;
				  if(prev == min(block,sizelist2))
				  {
					  list1_start++;
					  list2_start++;
					  break;
				  }
			  }

			  if(g_col_index[prev] == g_col_index[x])
			  {
				  total++;
				  list1_start++;
				  list2_start++;
				  break;
			  }
		  }
	  }
	  else
	  {
		  list1_start++;
		  list2_start++;
	  }

  //}

  return total; //return total triangles found by each thread...
}


__device__ int InterpolationIntersection(int src, int dst, int *g_col_index , int *g_row_ptr )
{
  int total = 0 ;
  int list1_start = g_row_ptr[src];
  int list1_end = g_row_ptr[src+1];
  int list2_start = g_row_ptr[dst];
  int list2_end = g_row_ptr[dst+1];

  //int sizelist2 = list2_end - list2_start;
  
  for( int x=list1_start; x<list1_end; x++)
  {
	  int lo = list2_start, hi = (list2_end-1);

	  while(lo <= hi && g_col_index[x] >= g_col_index[lo] && g_col_index[x] <= g_col_index[hi])
	  {
		  if(lo == hi)
		  {
			  if( g_col_index[lo] == g_col_index[x]) total++;
			  break;
		  }

		  int pos = lo + ((g_col_index[x] - g_col_index[lo]) * (hi-lo) / (g_col_index[hi]-g_col_index[lo]));
		  if( g_col_index[pos] == g_col_index[x])
			  total++;
		  if(g_col_index[pos] < g_col_index[x])
			  lo = pos + 1;
		  else
			  hi = pos - 1;
	  }
  }

  return total; //return total triangles found by each thread...
}


//--------------------------main kernel -------------------------------

__global__ void Find_Triangle_TwoPointer(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
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
      total = TwoPointerIntersection(id, g_col_index[i], g_col_index, g_row_ptr );
      //g_sum = g_sum + total;
      atomicAdd(&g_sum[0],total);
      }
    }
  }
}
__global__ void Find_Triangle_Binary(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
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
      total = BinaryIntersection(id, g_col_index[i], g_col_index, g_row_ptr );
      //g_sum = g_sum + total;
      atomicAdd(&g_sum[0],total);
      }
    }
  }
}
__global__ void Find_Triangle_Jump(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id 
  //__syncthreads();  // thread barrier

  if (id < vertex) // only number of vertex thread executed ...
  {
    for (int i = g_row_ptr[id] ; i < g_row_ptr[id+1] ; i++)
    {
      int total = 0;
     //******CALLED INTERSECTION FUNCTION ************
     
      if (id < g_col_index[i])
      {
      total = JumpIntersection(id, g_col_index[i], g_col_index, g_row_ptr );
      //g_sum = g_sum + total;
      atomicAdd(&g_sum[0],total);
      }
    }
  }
}
__global__ void Find_Triangle_Interpolation(int *g_col_index, int *g_row_ptr, int vertex, int edge ,int *g_sum )
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
      total = InterpolationIntersection(id, g_col_index[i], g_col_index, g_row_ptr );
      //g_sum = g_sum + total;
      atomicAdd(&g_sum[0],total);
      }
    }
  }
}


int main(int argc, char *argv[])
{
  cudaEvent_t start1,start2,start3,start4, stop1,stop2,stop3,stop4;
  cudaEventCreate(&start1);
  cudaEventCreate(&start2);
  cudaEventCreate(&start3);
  cudaEventCreate(&start4);
  cudaEventCreate(&stop1);
  cudaEventCreate(&stop2);
  cudaEventCreate(&stop3);
  cudaEventCreate(&stop4);

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

  cudaEventRecord(start1);
  Find_Triangle_TwoPointer<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
  cudaEventRecord(stop1);
  cudaDeviceSynchronize();
  cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
  int Triangle_TwoPointer = sum[0];

  cudaEventRecord(start2);
  Find_Triangle_Binary<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
  cudaEventRecord(stop2);
  cudaDeviceSynchronize();
  cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
  int Triangle_Binary = sum[0];

  cudaEventRecord(start3);
  Find_Triangle_Jump<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
  cudaEventRecord(stop3);
  cudaDeviceSynchronize();
  cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
  int Triangle_Jump = sum[0];

  cudaEventRecord(start4);
  Find_Triangle_Interpolation<<<nblocks,BLOCKSIZE>>>(g_col_index,g_row_ptr,vertex,edge,g_sum);
  cudaEventRecord(stop4);
  cudaDeviceSynchronize();


  
  //***********TAKE DATA FROM GPU TO CPU************
  cudaMemcpy(sum,g_sum,sizeof(int)*1,cudaMemcpyDeviceToHost);
  int Triangle_Interpolation = sum[0];
  //printf(" Total Triangles : %d\n ", Triangle);
  
  //*********PRINTING THE DATA *******************
  //printf("%d, ",edge);
  //printf("%d, ",triangle);
  //printf("%d, ",triangle-Triangle);
  
  //*******CHECKING DATA CORRECT OR NOT ****************
  //if (//triangle-Triangle == 0)
  //{ //printf("Correct, "); }
  //else
  //{ //printf("Not Correct, "); }

  cudaEventSynchronize(stop1);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  printf("\nTwo Pointer : %.4f sec ",milliseconds/1000);
  printf("\tTriangle TwoPointer : %d, ",Triangle_TwoPointer);
  
  cudaEventSynchronize(stop2);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start2, stop2);
  printf("\nBinary Search : %.4f sec ",milliseconds/1000);
  printf("\tTriangle Binary : %d, ",Triangle_Binary);
  
  cudaEventSynchronize(stop3);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start3, stop3);
  printf("\nJump Search : %.4f sec ",milliseconds/1000);
  printf("\tTriangle Jump : %d, ",Triangle_Jump);
  
  cudaEventSynchronize(stop4);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start4, stop4);
  printf("\nInterpolation : %.4f sec",milliseconds/1000);
  printf("\tTriangle Interpolation : %d, ",Triangle_Interpolation);
  //********** FREE THE MEMORY BLOCKS *****************
  free(col_index);
  free(row_ptr);
  cudaFree(g_col_index);
  cudaFree(g_row_ptr);
	printf("\n");
  return 0;
}
