#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>
#include<math.h>

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

  int total_tri = 0 ;
  int list1_start = d_row_ptr[id];
  int list1_end = d_row_ptr[id+1];
  int list2_start = d_row_ptr[index];
  int list2_end = d_row_ptr[index+1];
  //printf("\nList1-start :%d  ",list1_start);
  //printf("\nList1-end :%d  ",list2_end);
  //printf("\nList2-start :%d  ",list2_start);
  //printf("\nList2-end :%d  ",list2_end);

  int sizelist2 = list2_end - list2_start;
  //printf(" sizelist2 : %d  ",sizelist2);
  int step = (int)floor(sqrtf(sizelist2));
/*  
  for (int x=list1_start; x<list1_end; x++)
  {
	  int prev = list2_start;
	  int block = step+list2_start;
	  //printf("X:%d ",d_col_index[x]);

	  while( d_col_index[prev] < d_col_index[x] && prev < list2_end)
	  {
		  prev = block;
		  block = block + step;
		  if (block > list2_end)
		  {
			  block = list2_end-1;
			  break;
		  }
	  }
	  while(d_col_index[prev] < d_col_index[x] )
	  {
		  prev = prev+1;
		  if(prev > block)
		  {
			  //prev = list2_end;
			  break;
		  }
	  }
	  if(d_col_index[prev] == d_col_index[x])
	  {
		  total_tri = total_tri+1 ;
		  //printf("\n X: : %d , Prev : %d ",d_col_index[x],d_col_index[prev]);
		  //break;
	  }
  }
/*
  for (int x = list1_start; x < list1_end; x++)
  {
	  int start = list2_start;
	  int end = step+list2_start;
	  while( d_col_index[start] < d_col_index[x] && start < list2_end)
	  {
		  start = end;
		  end = end + step;
		  if( end > list2_end)
		  {
			  end = list2_end;
			  break;
		  }
	  }
	  for (int y=start; y<end; y++)
	  {
		  if(d_col_index[y] == d_col_index[x])
		  {
			  total_tri++;
		  }
	  }
  }
  
  return total_tri; //return total triangles found by each thread...
*/
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
  //int id = threadIdx.x;
  //__syncthreads();  // thread barrier

  if (id < total_v_in_partitions) // only number of vertex thread executed ...
  {
    for (int i = d_row_ptr[id] ; i < d_row_ptr[id+1] ; i++)
    {
      int total = 0;
     //******CALLED INTERSECTION FUNCTION ************

      //if (id < d_col_index[i])
      //{
	  //printf("\nsrc: %d , Dst:%d ",d_vertex_arr[id],d_col_index[i]);
      total = intersection(d_vertex_arr[id], d_col_index[i], d_col_index, d_row_ptr, d_vertex_arr ,v_pos ,id ,total_v_in_partitions);
      printf("\n edge(%d , %d) : %d",d_vertex_arr[id], d_col_index[i],total );
      //printf("\ntotal : %d",total);
      //g_sum = g_sum + total;
      atomicAdd(&d_sum[0],total);
      //}
    }
  }
}

int main(int argc, char *argv[])
{
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  float total_kernel_time = 0.0 ;
  float total_time = 0.0;
  clock_t start,end;
	double cpu_time_used;
  //------initilization of variables------------//
  int  data = 0 , pos = 0  , count = 0 , Total_Triangle = 0;
  int edge;
  int v_pos, rp_pos, ci_pos;

  char *argument3 = argv[3]; //take argument from terminal and initilize
  int vertex=atoi(argument3);

  char *argument4 = argv[4];
  int no_partitions = atoi(argument4);

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

  part_ptr[0]=count;

  for (int i=0; i<=vertex; i++)
  {
    fscanf(file,"%d", &data);
    data_array[i] = data;
  }
  fclose(file);

  //-----------------start sapration of partitons--------//
  start = clock();
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
  end = clock();
 	cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC;
 	printf(" \nsapration time : %f Sec  ",cpu_time_used);
	//total = total + cpu_time_used;
  //printf("\npart_ptr :");
  //for (int i=0; i<=no_partitions; i++) { printf("%d ",part_ptr[i]); }
  //printf("\npart_index :");
  //for (int i=0; i<vertex; i++) { printf("%d ",part_index[i]); }

  //----------------open directed CSR file and load into array----------//
  start = clock();
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
   printf("  No. of Edges : %d ",edge*2);
   //printf(" %d,",edge*2);
   //printf("\nrow_ptr : ");
   //for(int i=0; i<=vertex; i++) { printf("%d ",row_ptr[i]);}
   //printf("\ncol_index : ");
   //for(int j=0; j<edge; j++) { printf("%d ",col_index[j] );}
	end = clock();
 	cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC;
 	printf("    CSR load time : %f Sec\n",cpu_time_used);
	 //total = total + cpu_time_used;
   //---------------------make input for GPU--------------//


  printf("\n Partition      V_list_time   master_vertex  proxy_vertex  new_CSR_time   kernel_Time  total time  Triangle \n");
  for(int i=0; i<no_partitions; i++)
  {
	  start = clock();
     int *new_col_index;
    new_col_index= (int *) malloc(sizeof(int)*NUM_EDGES);

    int *new_row_ptr;
    new_row_ptr = (int *) malloc(sizeof(int)*NUM_VERTICES);

    int *vertex_arr;
    vertex_arr = (int *) malloc(sizeof(int)*NUM_VERTICES);

	  //--------make vertex number array ---------------//
    int total_v_in_partitions = part_ptr[i+1] - part_ptr[i];
    printf("  %d\t",i); //print parttitions number
    v_pos = 0 , rp_pos = 1, ci_pos = 0;
    new_row_ptr[0] = 0;
   // printf("\nvertex_arr :");
    for (int j=part_ptr[i]; j<part_ptr[i+1]; j++)
    {
      vertex_arr[v_pos] = part_index[j];
      //printf(" %d",vertex_arr[v_pos]);
      v_pos++;
    }
    end = clock();
 	cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC;
 	printf("\t %f Sec",cpu_time_used);
	 //total = total + cpu_time_used;
    //--------------make new_row_ptr and new_col_index------------//
    start = clock();

    printf("\t%d\t",total_v_in_partitions);
    int counter = 0;
    //float kernel_time = 0;
    for(int p=0; p<v_pos; p++)
    //for(int p=0; p<total_v_in_partitions; p++)
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
              //counter++;
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
              //counter++;
              break;
            }
          }
          //new_row_ptr[rp_pos] = ci_pos;
          //rp_pos++;
          if(flag == 0)
          {
            if(counter <= total_v_in_partitions)
            {
              //printf(" --------hello --------");
              new_col_index[ci_pos] = neighbour ;
              ci_pos++;
              vertex_arr[v_pos] = neighbour;
              v_pos++;
            }
            else
            {
              //printf(" --------hello --------");
              new_col_index[ci_pos] = neighbour ;
              ci_pos++;

            }
          }

      }
      new_row_ptr[rp_pos] = ci_pos;
      rp_pos++;
    }
    printf("\t %d ",v_pos-total_v_in_partitions);
    end = clock();
 	cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC;
 	printf("\t %4f Sec",cpu_time_used);
	 //total = total + cpu_time_used;
    //printf("\nvertex_arr : ");
    //for(int j=0; j<v_pos; j++){printf("%d ",vertex_arr[j]);}
    //printf("\nnew_row_ptr : ");
    //for(int j=0; j<rp_pos; j++){printf("%d ",new_row_ptr[j]);}
    //printf("\nnew_col_index : ");
    //for(int j=0; j<ci_pos; j++){printf("%d ",new_col_index[j]);}

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
    printf("    %.4f sec",millisecondsG/1000);
    total_kernel_time = total_kernel_time + millisecondsG/1000;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("  %.4f sec",milliseconds/1000);
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
  free(col_index);
  free(row_ptr);

return 0;
}
