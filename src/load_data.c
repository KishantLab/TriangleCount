#include<stdio.h>
#include<stdlib.h>

#define NUM_VERTICES 999999
#define NUM_EDGES 9999999


int main(int argc, char *argv[])
{
  //------initilization of variables------------//
  int Total_Triangle = 0;

  int v_pos, rp_pos, ci_pos, total_v_in_partitions;

  char *argument2 = argv[2]; //take argument from terminal and initilize
  int vertex=atoi(argument2);

  char *argument3 = argv[3]; //take argument from terminal and initilize
  int edge=atoi(argument3);

  char *argument4 = argv[4];
  int no_partitions = atoi(argument4);

  int *new_col_index;
  new_col_index= (int *) malloc(sizeof(int)*NUM_EDGES);

  int *new_row_ptr;
  new_row_ptr = (int *) malloc(sizeof(int)*NUM_VERTICES);

  int *vertex_arr;
  vertex_arr = (int *) malloc(sizeof(int)*NUM_VERTICES);

  int *v_pos_arr;
  v_pos_arr = (int *) malloc(sizeof(int)*no_partitions+1);

  int *rp_pos_arr;
  rp_pos_arr = (int *) malloc(sizeof(int)*no_partitions+1);

  int *ci_pos_arr;
  ci_pos_arr = (int *) malloc(sizeof(int)*no_partitions+1);

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
      total_v_in_partitions_arr[i+1] = ci_pos;
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
  return 0;
}
