#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define NUM_VERTICES 9999999999
#define NUM_EDGES 9999999999

int main(int argc, char *argv[])
{
	//------initilization of variables------------//
	int  data = 0 , pos = 0  , count = 0;
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

	printf("\n Arrays Created ............");

	//------------open partioton file------------//
	FILE *file1,*file2,*file3;
	file1 = fopen(argv[1],"r");

	if (file1 == NULL)
	{
		printf("File NOt opened");
		exit(0);
	}

	printf("\nPartitions File Opened ...............");
	part_ptr[0]=count;

	for (int i=0; i<=vertex+1; i++)
	{
		fscanf(file1,"%d", &data);
		data_array[i] = data;
	}
	fclose(file1);

	printf("\nData Loading Successfull ...............");

	//-----------------start sapration of partitons--------//

	printf("\nSapration Started ...............");
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

		//printf("\tcount: %d",count);
	}

	part_ptr[no_partitions+1] = count;
	free(data_array);

	printf("\nSapration Completed ..................");

	//----------------open directed CSR file and load into array----------//
	file2 = fopen(argv[2],"r");
	if(file2 == NULL)
	{
		printf("file not opened\n");
		exit(0);
	}
	else
	{
		printf("\nCSR file Opened..............");
		fscanf(file2 , "%d", &edge);

		for(int i=0; i<=vertex+1; i++)
		{
			fscanf(file2, "%d", &data);
			row_ptr[i]=data;

		}
		for(int j=0; j<edge; j++)
		{

			fscanf(file2,"%d", &data);
			col_index[j]=data;

		}
	}
	printf("   No. of Edges : %d ",edge*2);
	fclose(file2);

	printf("\nCSR Loading Successfull............");
	//---------------------make input for GPU--------------//
	file3 = fopen(argv[5],"w");
	printf("\nPartition Started .....");
	for(int i=0; i<no_partitions; i++)
	{
		//printf("\nstart P : %d\n" ,i);
		int *new_col_index;
		new_col_index= (int *) malloc(sizeof(int)*edge);

		int *new_row_ptr;
		new_row_ptr = (int *) malloc(sizeof(int)*vertex);

		int *vertex_arr;
		vertex_arr = (int *) malloc(sizeof(int)*vertex);

		//--------make vertex number array ---------------//
		int total_v_in_partitions = part_ptr[i+1] - part_ptr[i];
		//printf("  %d\t",i); //print parttitions number
		v_pos = 0 , rp_pos = 1, ci_pos = 0;
		new_row_ptr[0] = 0;
		for (int j=part_ptr[i]; j<part_ptr[i+1]; j++)
		{
			vertex_arr[v_pos] = part_index[j];
			v_pos++;
		}


		//--------------make new_row_ptr and new_col_index------------//

		//printf("\t%d\t",total_v_in_partitions);
		int counter = 0;
		for(int p=0; p<v_pos; p++)
		{
			int vertex_no = vertex_arr[p];
			counter++;

			for(int k=row_ptr[vertex_no]; k<row_ptr[vertex_no+1]; k++)
			{

				//printf(" K:%d ",k);

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
		printf("\n Partition %d is completed file Writing started .........",i);
		fprintf(file3,"%d ",v_pos);
		//printf("\n v_pos:%d" , v_pos);
		fprintf(file3,"%d ",rp_pos);
		//printf("\n rp_pos:%d" , rp_pos);
		fprintf(file3,"%d ",ci_pos);
		//printf("\n ci_pos:%d" , ci_pos);
		fprintf(file3,"%d ",total_v_in_partitions);
		//printf("\n total_v_in_partitions:%d" , total_v_in_partitions);
		fprintf(file3,"\n");
		//printf("\nArray writting started ........\n");

		//printf("\nVertex Array: ");
		for(int j=0; j<v_pos; j++)
		{
			fprintf(file3,"%d ",vertex_arr[j]);
			//printf("%d ",vertex_arr[j]);
		}
		fprintf(file3,"\n");
		//printf("\nRow Ptr Array: ");
		for(int j=0; j<rp_pos; j++)
		{
			fprintf(file3,"%d ",new_row_ptr[j]);
			//printf("%d ",new_row_ptr[j]);
		}
		fprintf(file3,"\n");
		//printf("\nCol Index Array:");
		for(int j=0; j<ci_pos; j++)
		{
			fprintf(file3,"%d ",new_col_index[j]);
			//printf("%d ",new_col_index[j]);
		}
		fprintf(file3,"\n");
		//printf("\nProxy vertex Are: %d ",v_pos-total_v_in_partitions);
		//printf("\nFile Writing Successfull for P:%d\n ",i);
		//}
}

//free(part_ptr);
//free(part_index);
//free(col_index);
//free(row_ptr);

return 0;
}
