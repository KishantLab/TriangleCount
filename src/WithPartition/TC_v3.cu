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
/*
  for(int i=id; i<v_pos; i++)
  {
    if( d_vertex_arr[i] == dst){index = i ;break;}
  }

*/
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

/*
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
			  if ( d_col_index[mid] < d_col_index[i] ){ low = mid; }
			  else if ( d_col_index[mid] > d_col_index[i] ){ high = mid; }
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
  {	  for (int i=pointer2_start; i<pointer2_end; i++)
	  //while (pointer2_start <= pointer2_end)
          {
                  int low = pointer1_start;
                  int high = pointer1_end;
                  int mid = 0;
                  while (high - low > 1)
                  {
                          mid = (high + low)/2;
                          if ( d_col_index[mid] < d_col_index[i] ){ low = mid; }
                          else if ( d_col_index[mid] > d_col_index[i] ){ high = mid; }
                          else
			  {
			  	total++;
			  	break;
			  }
                  }
		  //pointer2_start++;
         }
  }
*/
  return total; //return total triangles found by each thread...
}

__global__ void Find_Triangle(int *d_col_index, int *d_row_ptr, int *d_vertex_arr,int total_v_in_partitions, int v_pos, int rp_pos, int ci_pos, int *d_sum )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
  //__syncthreads();  // thread barrier

  if (id < total_v_in_partitions) // only number of vertex thread executed ...
  {
    for (int i = d_row_ptr[id] ; i < d_row_ptr[id+1] ; i++)
    {
      int total = 0;
     //******CALLED INTERSECTION FUNCTION ************
      total = intersection(d_vertex_arr[id], d_col_index[i], d_col_index, d_row_ptr, d_vertex_arr ,v_pos ,id ,total_v_in_partitions);
      //printf("\n edge(%d , %d) : %d",d_vertex_arr[id], d_col_index[i],total );
      //printf("\ntotal : %d",total);
      atomicAdd(&d_sum[0],total);
    }
  }
}

int main(int argc, char *argv[])
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_kernel_time = 0.0 ;

  //clock_t start,end;
  //double cpu_time_used, total;
  //------initilization of variables------------//
  int  data = 0 , pos = 0  , count = 0 , Total_Triangle = 0;
  int edge;
  int v_pos, rp_pos, ci_pos;

  char *argument3 = argv[3]; //take argument from terminal and initilize
  int vertex=atoi(argument3);

  char *argument4 = argv[4];
  int no_partitions = atoi(argument4);

  cudaStream_t stream[no_partitions]; //declare stream for Device 
  
  //printf(", %d,",vertex);
  //char *part_file = argv[1];
  //printf("\n%s",part_file);

  //char *file_name = argv[2];
  //printf("\t%s",file_name);

  //-------------declare arrays-----------------//
  int *data_array;
  data_array = (int *)malloc(sizeof(int)*NUM_VERTICES);

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

  //printf("\npart_ptr :");
  //for (int i=0; i<=no_partitions; i++) { printf("%d ",part_ptr[i]); }
  //printf("\npart_index :");
  //for (int i=0; i<vertex; i++) { printf("%d ",part_index[i]); }

  //----------------open directed CSR file and load into array----------//
  file = fopen(argv[2],"r");
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
   printf("   No. of Edges : %d\n ",edge*2);

   //---------------------make input for GPU--------------//

   printf("\n Partition  total_v_in_partition  Triangle  kernel_Time \n");

  for(int i=0; i<no_partitions; i++)
  {
	  //printf("start P : %d\n" ,i);
    int *new_col_index;
    new_col_index= (int *) malloc(sizeof(int)*edge);

    int *new_row_ptr;
    new_row_ptr = (int *) malloc(sizeof(int)*vertex);

    int *vertex_arr;
    vertex_arr = (int *) malloc(sizeof(int)*vertex);
	
    //start = clock();
    //--------make vertex number array ---------------//
    int total_v_in_partitions = part_ptr[i+1] - part_ptr[i];
    printf("  %d\t",i); //print parttitions number
    v_pos = 0 , rp_pos = 1, ci_pos = 0;
    new_row_ptr[0] = 0;
    for (int j=part_ptr[i]; j<part_ptr[i+1]; j++)
    {
      vertex_arr[v_pos] = part_index[j];
      v_pos++;
    }
    //--------------make new_row_ptr and new_col_index------------//
 
    printf("\t%d\t",total_v_in_partitions);
    int counter = 0;
    for(int p=0; p<v_pos; p++)
    {
      int vertex_no = vertex_arr[p];
      counter++;
      //printf("\n test %d \n",p);
      for(int k=row_ptr[vertex_no]; k<row_ptr[vertex_no+1]; k++)
      {
        //printf("\n test %d \n",k);
        //counter++;
          int neighbour = col_index[k];
          //printf("\n vertex_no : %d neighbour : %d ",vertex_no,neighbour);
          int low = 0 , high = total_v_in_partitions, mid;
          int flag = 0;
          //printf("\n low %d high %d ",low,high);
          //printf("\n total_v_in_partitions : %d ",total_v_in_partitions);
          while(high - low > 1)
          {
            mid = (high + low )/2;
            //printf(" mid %d",mid);
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
    //printf("\nvertex_arr : ");
    //for(int j=0; j<v_pos; j++){printf("%d ",vertex_arr[j]);}
    //printf("\nnew_row_ptr : ");
    //for(int j=0; j<rp_pos; j++){printf("%d ",new_row_ptr[j]);}
    //printf("\nnew_col_index : ");
    //for(int j=0; j<ci_pos; j++){printf("%d ",new_col_index[j]);}
	
    //end = clock();
    //cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC;
    //printf(" \n Total Time Taken By CPU : %f Sec. ",cpu_time_used);
    //--------------------Launch the kernel-------------------//

    cudaStreamCreate(&stream[i]);  //create stream for Device

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
	
    //cudaStreamCreate(&stream[i]);

    int nblocks = ceil((float)total_v_in_partitions / BLOCKSIZE);

    //--------copy data from host to device --------------//
    cudaMemcpyAsync(d_col_index,new_col_index,sizeof(int)*ci_pos,cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(d_row_ptr,new_row_ptr,sizeof(int)*rp_pos,cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(d_vertex_arr,vertex_arr,sizeof(int)*v_pos,cudaMemcpyHostToDevice,stream[i]);

    //---------------------------kernel callled------------------//
    cudaEventRecord(start);
    //cudaDeviceSynchronize();
    //printf("kernel called \n");
    Find_Triangle<<<nblocks,BLOCKSIZE,0,stream[i]>>>(d_col_index,d_row_ptr,d_vertex_arr,total_v_in_partitions,v_pos,rp_pos,ci_pos,d_sum);
    cudaEventRecord(stop);
	//printf("Async start\n");
    cudaMemcpyAsync(sum,d_sum,sizeof(int)*1,cudaMemcpyDeviceToHost,stream[i]);
    int Triangle = sum[0];
    //atomicAdd(&sum[0],Triangle);
    printf("\t%d\t" , Triangle);
    Total_Triangle = Total_Triangle + Triangle ;
	//printf("start next P\n");
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%.4f sec\n",milliseconds/1000);
    total_kernel_time = total_kernel_time + milliseconds/1000;
	
    //printf("%f Sec \n",cpu_time_used);
    //cudaStreamDestroy(stream[i]);
    free(new_row_ptr);
    free(new_col_index);
    free(vertex_arr);
    //cudaFree(d_row_ptr);
    //cudaFree(d_col_index);
    //cudaFree(d_vertex_arr);
    //cudaStreamDestroy(stream[i]);


  }
  printf("\nTotal Triangle : %d ",Total_Triangle );
  printf("\t Total Kernel Time : %.4f sec \n",total_kernel_time);
  free(part_ptr);
  free(part_index);
  free(col_index);
  free(row_ptr);

return 0;
}
