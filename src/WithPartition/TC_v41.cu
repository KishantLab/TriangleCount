#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

#define NUM_VERTICES 9999999999
#define NUM_EDGES 9999999999
#define BLOCKSIZE 1024

__device__ int intersection(int src, int dst, int *d_col_index, int *d_row_ptr, int *d_vertex_arr ,int v_pos,int id ,int total_v_in_partitions)
{
  //******initilized Variables *****************
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

__global__ void Find_Triangle(int *d_col_index, int *d_row_ptr, int *d_vertex_arr,int total_v_in_partitions, int v_pos, int rp_pos, int ci_pos, int *d_sum )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id

  if (id < total_v_in_partitions) // only number of vertex thread executed ...
  {
    for (int i = d_row_ptr[id] ; i < d_row_ptr[id+1] ; i++)
    {
      int total = 0;

      //******CALLED INTERSECTION FUNCTION ************
      total = intersection(d_vertex_arr[id], d_col_index[i], d_col_index, d_row_ptr, d_vertex_arr ,v_pos ,id ,total_v_in_partitions);
      atomicAdd(&d_sum[0],total);
    }
  }
}

int sapration (int *part_ptr, int *part_index, int no_partitions, int vertex, FILE *file)
{
  int *data_array;
  data_array = (int *)malloc(sizeof(int)*NUM_VERTICES);
  int pos = 0, count = 0, data = 0;

  if (file == NULL)
  {
	  printf("File NOt opened");
	  exit(0);
  }

  part_ptr[0]=count;

  for (int i=0; i<=vertex; i++)
  {
    fscanf(file,"%d", &data);
    data_array[i] = data;
  }
  fclose(file);

  //-----------------start sapration of partitons--------//
  for (int i=0; i<no_partitions; i++)
  {
    for (int j=0; j<vertex; j++)
    {
      if(i == data_array[j])
      {
        part_index[pos] = j+1;
        pos++;
        count++;
      }
    }
    part_ptr[i+1] = count;
  }

  part_ptr[no_partitions+1] = count;
  free(data_array);
  return 0;
}

//***********************************LOAD CSR MATRIX FROM FILE******************//
int loadcsr(int *row_ptr, int *col_index, int vertex, int edge, FILE *file)
{
  int data = 0;
  if(file == NULL)
  {
    printf("file not opened\n");
    exit(0);
  }
  else
  {
    fscanf(file , "%d", &edge);

    for(int i=0; i<=vertex; i++)
    {
      fscanf(file, "%d", &data);
      row_ptr[i]=data;
    }
    for(int j=0; j<edge; j++)
    {

        fscanf(file,"%d", &data);
        col_index[j]=data;
     }
   }
   printf("   No. of Edges : %d ",edge*2);
   return 0;
}

int make_partitions(int *part_ptr, int *part_index, int *row_ptr, int *col_index, int *new_row_ptr, int *new_col_index, int *vertex_arr, int *v_pos, int *rp_pos, int *ci_pos, int vertex, int edge, int no_partitions)
{
  for(int i=0; i<no_partitions; i++)
  {
    //--------make vertex number array ---------------//

    int total_v_in_partitions = part_ptr[i+1] - part_ptr[i];

    printf("%d\t",i); //print parttitions number
    *v_pos = 0 , *rp_pos = 1, *ci_pos = 0;
    new_row_ptr[0] = 0;
    for (int j=part_ptr[i]; j<part_ptr[i+1]; j++)
    {
      vertex_arr[*v_pos] = part_index[j];
      *v_pos++;
    }

    //--------------make new_row_ptr and new_col_index------------//

    printf("\t%d\t",total_v_in_partitions);
    int counter = 0;
    for(int p=0; p<v_pos; p++)
    {
      int vertex_no = vertex_arr[p];
      counter++;

      for(int k=row_ptr[vertex_no]; k<row_ptr[vertex_no+1]; k++)
      {
          int neighbour = col_index[k];
          int low = 0 , high = total_v_in_partitions, mid;
          int flag = 0;
          while(high - low > 1)
          {
            mid = (high + low )/2;
            if (vertex_arr[mid] < neighbour) { low = mid; }
            else if (vertex_arr[mid] > neighbour) { high = mid; }
            else if (vertex_arr[mid] == neighbour)
            {
              new_col_index[ci_pos] = neighbour ;
              ci_pos++;
              flag++;
              break;
            }
          }
          for(int q=total_v_in_partitions; q<v_pos; q++)
          {
            if( vertex_arr[q] == neighbour)
            {
              new_col_index[ci_pos] = neighbour ;
              ci_pos++;
              flag++;
              break;
            }
          }
          if(flag == 0)
          {
            if(counter <= total_v_in_partitions)
            {
              new_col_index[ci_pos] = neighbour ;
              ci_pos++;
              vertex_arr[v_pos] = neighbour;
              v_pos++;
            }
            else
            {
              new_col_index[ci_pos] = neighbour ;
              ci_pos++;
            }
          }
      }
      new_row_ptr[rp_pos] = ci_pos;
      rp_pos++;
    }
    printf("\t %d ",v_pos-total_v_in_partitions);
  return 0;
}

int write_to_file(int *vertex_arr, int *new_row_ptr, int *new_col_index, int v_pos, int rp_pos, int ci_pos, FILE *file)
{
  for(int j=0; j<v_pos; j++){fprintf(file,"%d",vertex_arr[j]);}fprintf(file,"\n");
  for(int j=0; j<rp_pos; j++){fprintf(file,"%d ",new_row_ptr[j]);}fprintf(file,"\n");
  for(int j=0; j<ci_pos; j++){fprintf(file,"%d ",new_col_index[j]);}fprintf(file,"\n");
}

//***************************Main FUNCTION start********************************//

int main(int argc, char *argv[])
{
  float total_kernel_time = 0.0 ;
  float total_time = 0.0;
  clock_t start,end;
  double cpu_time_used;

  //------initilization of variables------------//
  int Total_Triangle = 0;
  int edge;
  int v_pos, rp_pos, ci_pos;

  char *argument3 = argv[3]; //take argument from terminal and initilize
  int vertex=atoi(argument3);

  char *argument4 = argv[4];
  int no_partitions = atoi(argument4);

  //-------------declare arrays-----------------//
  int *part_ptr;
  part_ptr = (int *)malloc(sizeof(int)*no_partitions+1);

  int *part_index;
  part_index = (int *)malloc(sizeof(int)*NUM_VERTICES);

  int *col_index;
  col_index= (int *) malloc(sizeof(int)*NUM_EDGES);

  int *row_ptr;
  row_ptr = (int *) malloc(sizeof(int)*NUM_VERTICES);



  //------------open partioton file------------//
  FILE *file;

  file = fopen(argv[1],"r");
  sapration(part_ptr, part_index, no_partitions, vertex,file);
  fclose(argv[1]);

  file = fopen(argv[2],"r");
  loadcsr(row_ptr, col_index, vertex, edge,file);
  fclose(argv[3]);

  int *new_col_index;
  new_col_index= (int *) malloc(sizeof(int)*edge);

  int *new_row_ptr;
  new_row_ptr = (int *) malloc(sizeof(int)*vertex);

  int *vertex_arr;
  vertex_arr = (int *) malloc(sizeof(int)*vertex);

   //---------------------make input for GPU--------------//
   file = fopen(argv[5],"r");
   if (file == NULL)
   {
	file = fopen(argv[5],"w");
	for (int i=0; i<no_partitions; i++)
     {
       make_partitions(part_ptr, part_index, row_ptr, col_index, new_row_ptr, new_col_index, vertex_arr, v_pos, rp_pos, ci_pos, vertex, edge);

	   write_to_file(vertex_arr,new_row_ptr,new_col_index,v_pos,rp_pos,ci_pos,file)

     }
   }
   else{
     for (int i=0; i<no_partitions; i++)
     {
		for(int j=0; j<v_pos; j++){fscanf(file,"%d",&data); vertex_arr[j] = data; }
		for(int j=0; j<rp_pos; j++){fscanf(file,"%d ",&data); new_row_ptr[j] = data;}
		for(int j=0; j<ci_pos; j++){fscanf(file,"%d ",&data); new_col_index[j] = data;}
     }
   }
   printf("\n Partition      V_list_time   master_vertex  proxy_vertex  new_CSR_time   kernel_Time  total_time  Triangle \n");


    //--------------------Launch the kernel-------------------//

    //cudaStreamCreate(&stream[i]);  //create stream for Device

    int *d_col_index;  //GPU MEMORY ALOOCATION
    cudaMalloc(&d_col_index,sizeof(int)*ci_pos);

    int *d_vertex_arr;  //GPU MEMORY ALOOCATION
    cudaMalloc(&d_vertex_arr,sizeof(int)*v_pos);

    int *d_row_ptr;   // GPU MEMORY ALLOCATION
    cudaMalloc(&d_row_ptr,sizeof(int)*rp_pos);

    int *d_sum;
    int *sum;
    sum= (int *)malloc(sizeof(int)*1);
    cudaMalloc((void**)&d_sum,sizeof(int)*1);


    int nblocks = ceil((float)total_v_in_partitions / BLOCKSIZE);

     cudaEvent_t start, stop,startG,stopG;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startG);
    cudaEventCreate(&stopG);

    cudaEventRecord(start);
    //--------copy data from host to device --------------//

    cudaMemcpyAsync(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(d_row_ptr,new_row_ptr,sizeof(int)*rp_pos,cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(d_vertex_arr,vertex_arr,sizeof(int)*v_pos,cudaMemcpyHostToDevice,stream[i]);

    //---------------------------kernel callled------------------//

    cudaEventRecord(startG);

    Find_Triangle<<<nblocks,BLOCKSIZE,0,stream[i]>>>(d_col_index,d_row_ptr,d_vertex_arr,total_v_in_partitions,v_pos,rp_pos,ci_pos,d_sum);
    cudaEventRecord(stopG);

    int Triangle = sum[0];
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float millisecondsG = 0;
    cudaEventElapsedTime(&millisecondsG, startG, stopG);
    printf("    %.4f sec",millisecondsG/1000);
    total_kernel_time = total_kernel_time + millisecondsG/1000;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    %.4f sec",milliseconds/1000);
    total_time = total_time + milliseconds/1000;

    printf("\t%d\n" , Triangle);
    Total_Triangle = Total_Triangle + Triangle ;

    free(new_row_ptr);
    free(new_col_index);
    free(vertex_arr);
    //cudaFree(d_row_ptr);
    //cudaFree(d_col_index);
    //cudaFree(d_vertex_arr);
    //cudaStreamDestroy(stream[i]);


  }
  printf("\nTotal Triangle : %d ",Total_Triangle );
  printf("\t Total Kernel Time : %.4f sec",total_kernel_time);
  printf("\t Total Time : %f \n\n",total_time);
  free(part_ptr);
  free(part_index);
  free(col_index);
  free(row_ptr);

return 0;
}
