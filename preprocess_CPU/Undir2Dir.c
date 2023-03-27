#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#include "progressbar.hpp"

int main(int argc, char *argv[])
{

  char *argument3 = argv[3];
  unsigned long long int nopart=atoi(argument3);
  progressbar bar(10);
  unsigned long long int data=0;
  //-------------------------OPEN FILE FOR READING AND WRRITING-------------------------------#
  FILE *file;
  file = fopen(argv[1],"r");
  FILE *file1;
  file1 = fopen(argv[2],"w");
  if(file1 == NULL)
  {
    printf("File not opened");
    exit(0);
  }
  if (file == NULL)
  {
    printf("file not opened \n");
    exit(0);
  }
  else
  {
    //------------------------------DEG OF ALL VERTEX READING--------------------------#
    unsigned long long int in_deg_s=0;
    fscanf(file, "%llu", &in_deg_s);
    unsigned long long int *in_deg;
    in_deg= (unsigned long long int *) malloc(sizeof(unsigned long long int)*in_deg_s);
    bar.reset();
    printf("\nRead Degrees : ");
    bar.set_niter(in_deg_s);
    for(unsigned long long int i=0; i<in_deg_s; i++)
    {
      fscanf(file, "%llu" , &data);
      in_deg[i] = data;
      bar.update();
    }

    //----------------------------CONVERT UNDIRECTED PARTED ARRAY TO DIRECTED ARRAY-----//
    for(unsigned long long int z=0; z<nopart; z++)
    {
      //-------------------------------INITILIZATION AND READING DATA-------------------//
      unsigned long long int org_id_s=0, row_ptr_s=0, col_idx_s=0, t_ver=0;

      fscanf(file, "%llu", &org_id_s);
      fscanf(file, "%llu", &row_ptr_s);
      fscanf(file, "%llu", &col_idx_s);
      fscanf(file, "%llu", &t_ver);

      unsigned long long int *org_id;
      org_id= (unsigned long long int *) malloc(sizeof(unsigned long long int)*org_id_s);
      unsigned long long int *row_ptr;
      row_ptr= (unsigned long long int *) malloc(sizeof(unsigned long long int)*row_ptr_s);
      unsigned long long int *col_idx;
      col_idx= (unsigned long long int *) malloc(sizeof(unsigned long long int)*col_idx_s);
      unsigned long long int *row_ptr_Dir;
      row_ptr_Dir = (unsigned long long int *) malloc(sizeof(unsigned long long int)*row_ptr_s);
      unsigned long long int *col_index_Dir;
      col_index_Dir= (unsigned long long int *) malloc(sizeof(unsigned long long int)*col_idx_s);

      bar.reset();
      printf("\nRead Global ID : ");
      bar.set_niter(org_id_s);
      for(unsigned long long int i=0; i<org_id_s; i++)
      {
        fscanf(file, "%llu" , &data);
        org_id[i] = data ;
        bar.update();
      }
      bar.reset();
      printf("\nRead Row_ptr : ");
      bar.set_niter(row_ptr_s);
      for(unsigned long long int i=0; i<row_ptr_s; i++)
      {
        fscanf(file, "%llu" , &data);
        row_ptr[i] = data ;
        bar.update();
      }
      bar.reset();
      printf("\nRead col_idx : ");
      bar.set_niter(col_idx_s);
      for(unsigned long long int j=0; j<col_idx_s; j++)
      {
        fscanf(file, "%llu" , &data);
        col_idx[j] = data;
        bar.update();
      }
      //-------------------------------------------CONVERTING DATA INTO DIRECTED--------//
      // printf("Undirected Array\n");
      printf("\nTotal VERTEX : %llu\tTotal Edges : %llu\tTotal Mster Vertex : %llu",row_ptr_s,col_idx_s,t_ver);

      //START CONVERTING UNDIRECTED CSR TO DIRECTED CSR
      unsigned long long int pos = 0 ;
      row_ptr_Dir[0] = 0;
      //pos++;
      //#pragma omp parallel for
      bar.reset();
      printf("\nConverting Directed : ");
      bar.set_niter(row_ptr_s+col_idx_s);
      for (unsigned long long int i=0; i<row_ptr_s-1; i++)
      {
        //unsigned long long int deg_src = row_ptr[i+1] - row_ptr[i];
        unsigned long long int deg_src = in_deg[org_id[i]];
        //printf("\n Deg_src : %llu ",deg_src);
        for(unsigned long long int j=row_ptr[i]; j<row_ptr[i+1]; j++)
        {
          //unsigned long long int deg_dst = row_ptr[col_index[j]+1] - row_ptr[col_index[j]];
          unsigned long long int deg_dst = in_deg[org_id[col_idx[j]]];
          //printf("  Deg_dst : %llu ",deg_dst);
          if(deg_src < deg_dst)
          {
            col_index_Dir[pos] = col_idx[j];
    	       pos++;
          }
          else if(deg_src == deg_dst)
          {
            if(org_id[col_idx[j]] < org_id[i])
            {
              col_index_Dir[pos] = col_idx[j];
    	         pos++;
            }
          }
          bar.update();
          row_ptr_Dir[i+1] = pos;
        }
      }
      //Writing into Another File
      printf("\nTotal Edges : %llu\n", pos);

      fprintf(file1, "%llu ", t_ver);
      fprintf(file1, "%llu ", row_ptr_s);
      fprintf(file1, "%llu\n", pos);
      bar.reset();
      printf("\nWritting row_ptr : ");
      bar.set_niter(row_ptr_s);
      for(unsigned long long int i=0; i<row_ptr_s; i++)
      {
        fprintf(file1, "%llu ",row_ptr_Dir[i]);
        bar.update();
      }
      fprintf(file1,"\n");
      bar.reset();
      printf("\nWritting col_idx : ");
      bar.set_niter(pos);
      for(unsigned long long int j=0; j<pos; j++)
      {
        fprintf(file1, "%llu ", col_index_Dir[j]);
        bar.update();
      }
      fprintf(file1,"\n");
      free(org_id);
      free(col_idx);
      free(row_ptr);
      free(row_ptr_Dir);
      free(col_index_Dir);
    }
    free(in_deg);
  }
  fclose(file);
  fclose(file1);
  printf("\nDirected file Created Succesfull..\n");
  return 0;
}
