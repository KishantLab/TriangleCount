#include<stdio.h>
#include<stdlib.h>

#define NUM_VERTICES 9999999999
#define NUM_EDGES 9999999999

void main(int argc, char *argv[])
{
  //UNDIRECTED ARRAYS
  int *col_index;
  col_index= (int *) malloc(sizeof(int)*NUM_VERTICES);
  int *row_ptr;
  row_ptr = (int *) malloc(sizeof(int)*NUM_EDGES);

  //DIRECTED ARRAYS
  int *col_index_Dir;
  col_index_Dir= (int *) malloc(sizeof(int)*NUM_VERTICES);
  int *row_ptr_Dir;
  row_ptr_Dir = (int *) malloc(sizeof(int)*NUM_EDGES);

  int edge = 0 , data = 0 ;
  //int vertex = 3774768;
  
  char *argument3 = argv[3];
  int vertex=atoi(argument3);

  // OPENING OF FILE
  //READING DATA FROM FILE AND STORE IN ARRAY
  FILE *file;
  file = fopen(argv[1],"r");

  if (file == NULL)
  {
    printf("file not opened \n");
    exit(0);
  }
  else
  {
    fscanf(file, "%d", &edge);
    for(int i=0; i<=vertex+1; i++)
    {
      fscanf(file, "%d" , &data);
      row_ptr[i] = data ;
    }
    for(int j=0; j<edge; j++)
    {
      fscanf(file, "%d" , &data);
      col_index[j] = data;
    }
  }
  fclose(file);
  //printf("col-size Undir: %d\n",col_index_size);
  //printf("row-size Undir: %d\n",row_ptr_size);
	printf("Undirected Array\n");
  printf("Total Edges : %d\n", edge);
/*
  for(int i=0; i<vertex; i++)
  {
    printf("%d ",row_ptr[i]);
  }
  printf("\n");
  for(int j=0; j<edge; j++)
  {
    printf("%d ", col_index[j]);
  }
  printf("\n");
*/
  //START CONVERTING UNDIRECTED CSR TO DIRECTED CSR
  int pos = 0 ;
  row_ptr_Dir[0] = 0;
  //pos++;

  for (int i=0; i<=vertex; i++)
  {
    for (int j=row_ptr[i]; j<row_ptr[i+1]; j++)
    {
      //printf("%d " , row_ptr[i]);
      if (col_index[j]>i)
      {
        col_index_Dir[pos] = col_index[j];
        pos++;
      }
      row_ptr_Dir[i+1] = pos;
    }
  }


  //CALCULATING THE SIZE OF NEW ROW POINTER AND COLUMN col_index

  //Writing into Another File

  file = fopen(argv[2],"w");
	printf("Directed Array \n");
  printf("Total Edges : %d\n", pos);
/*
  for(int i=0; i<=vertex; i++)
  {
    printf("%d ",row_ptr_Dir[i]);
  }
  printf("\n");
  for(int j=0; j<pos; j++)
  {
    printf("%d ", col_index_Dir[j]);
  }
  printf("\n");

*/
  if(file == NULL)
  {
    printf("File not opened");
    exit(0);
  }
  else
  {
    fprintf(file, "%d\n", pos);
    for(int i=0; i<=vertex; i++)
    {
      fprintf(file, "%d ",row_ptr_Dir[i]);
    }
    fprintf(file,"\n");
    for(int j=0; j<pos; j++)
    {
      fprintf(file, "%d ", col_index_Dir[j]);
    }
  }

  fclose(file);
  printf("Directed file Created Succesfull..\n");
  free(col_index);
  free(row_ptr);
  free(row_ptr_Dir);
  free(col_index_Dir);
}
