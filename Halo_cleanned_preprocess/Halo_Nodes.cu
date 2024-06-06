#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <cusparse_v2.h>
#include <sys/time.h>
using namespace std;

#define BLOCK_SIZE 1024


__device__ __forceinline__ unsigned long long int Search (unsigned long long int skey , unsigned long long int *neb, unsigned long long int sizelist)
{
	unsigned long long int total = 0;
	if(skey < neb[0] || skey > neb[sizelist])
	{
		return 0;
	}
	else if(skey == neb[0] || skey == neb[sizelist])
	{
		return 1;
	}
	else
	{
		unsigned long long int lo=1;
		unsigned long long int hi=sizelist-1;
		unsigned long long int mid=0;
		while( lo <= hi)
		{
			mid = (hi+lo)/2;
			//printf("\nskey :%d , mid : %d ",skey,neb[mid]);
			if( neb[mid] < skey){lo=mid+1;}
			else if(neb[mid] > skey){hi=mid-1;}
			else if(neb[mid] == skey)
			{
				total++;
				break;
			}
		}
	}
	return total;
}

__global__ void find_halo_nodes(unsigned long long *d_row_ptr_G, unsigned long long *d_col_idx_G,
                                signed long long *d_halo_nodes, unsigned long long induced_nodes_s,
                                unsigned long long *d_induced_nodes, unsigned long long *d_pos)
{
    unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < induced_nodes_s)
    {
        unsigned long long node = d_induced_nodes[id];
        unsigned long long start = d_row_ptr_G[node];
        unsigned long long end = d_row_ptr_G[node + 1];
        for (unsigned long long i = start; i < end; i++)
        {
            unsigned long long flag = 0;
            for (unsigned long long j = 0; j < induced_nodes_s; j++)
            {
                if (d_col_idx_G[i] == d_induced_nodes[j])
                {
                    flag = 1;
                    break;
                }
            }
            if (flag != 1)
            {
                unsigned long long pos = atomicAdd(d_pos, 1);
                d_halo_nodes[pos] = d_col_idx_G[i];
            }
        }
    }
}

__global__ void find_halo_nodes1(unsigned long long *d_row_ptr_G, unsigned long long *d_col_idx_G,
                                signed long long *d_halo_nodes, unsigned long long induced_nodes_s,
                                unsigned long long *d_induced_nodes, unsigned long long *d_pos)
{
    unsigned long long bid = blockIdx.x;
    __shared__ unsigned long long k = 0;
    if (bid < induced_nodes_s)
    {
        unsigned long long node = d_induced_nodes[bid];
        unsigned long long start = d_row_ptr_G[node];
        unsigned long long end = d_row_ptr_G[node + 1];
        unsigned long long deg = end - start;
        for (unsigned long long i = threadIdx.x; i < deg; i+= BLOCK_SIZE)
        {
            // unsigned long long flag = 0;
            // unsigned long long pos = 0;
            unsigned long long flag = Search(d_col_idx_G[start+i], d_induced_nodes, induced_nodes_s-1);
            // for (unsigned long long j = 0; j < induced_nodes_s; j++)
            // {
            //     if (d_col_idx_G[start+i] == d_induced_nodes[j])
            //     {
            //         flag = 1;
            //         break;
            //     }
            // }
            // printf("%llu ", flag);
            if (flag == 0)
            {
                // pos = d_pos[0];
                // unsigned long long pos = atomicAdd(d_pos, 1);
                // unsigned long long k = 0;
                while (true) {
                    if(d_halo_nodes[k]!= -1){
                        k++;
          }else {
                d_halo_nodes[k] = d_col_idx_G[start+i];
            // pos = k;
            break;
          }
                }
                // d_halo_nodes[pos] = d_col_idx_G[start+i];
                // printf("%llu ", d_halo_nodes[pos]);
            }
        }
    }
}

void create_subgraph_with_halo(unsigned long long nopart, const unsigned long long *node_parts,
                               const unsigned long long *row_ptr_G,
                               const unsigned long long *col_idx_G,
                               const unsigned long long *in_deg,
                               const string &output_filename,
                               unsigned long long num_nodes,
                               unsigned long long num_edges)
{
    ofstream outfile(output_filename);
    if (!outfile.is_open())
    {
        cerr << "Error opening file: " << output_filename << endl;
        return;
    }

    for (unsigned long long value = 0; value < nopart; ++value)
    {
        cout << value << "th Partitions Constructions with halo nodes .." << endl;

        cout << "Finding Indices..." << endl;

        // Finding indices of the partition
        unsigned long long *indices = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
        unsigned long long indices_size = 0;
        for (long long int i = 0; i < num_nodes; ++i) {
            if (node_parts[i] == value) {
                indices[indices_size++] = i;
            }
        }
        outfile<<indices_size<<endl;

        unsigned long long induced_nodes_s = indices_size;
        unsigned long long *d_induced_nodes;
        cudaMalloc(&d_induced_nodes, induced_nodes_s * sizeof(unsigned long long));
        cudaMemcpy(d_induced_nodes, indices, induced_nodes_s * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        unsigned long long *d_pos;
        cudaMalloc(&d_pos, sizeof(unsigned long long));
        cudaMemset(d_pos, 0, sizeof(unsigned long long));

        signed long long *d_halo_nodes;
        cudaMalloc(&d_halo_nodes, num_edges * sizeof(signed long long));
        cudaMemset(d_halo_nodes, -1, num_edges * sizeof(signed long long));


        cout << "Indices: ";
        for (unsigned long long i = 0; i < indices_size; ++i) {
            cout << indices[i] << " ";
        }
        cout << endl;

        unsigned long long *d_row_ptr_G;
        unsigned long long *d_col_idx_G;
        cudaMalloc(&d_row_ptr_G, (num_nodes+1) * sizeof(unsigned long long));
        cudaMalloc(&d_col_idx_G, num_edges * sizeof(unsigned long long));
        cudaMemcpy(d_row_ptr_G, row_ptr_G, (num_nodes+1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx_G, col_idx_G, num_edges * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        unsigned long long grid_size_G = ((num_nodes+1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cout << "Number of induced nodes: " << induced_nodes_s << endl;
        struct timeval begin2, end2;
        gettimeofday(&begin2, 0);
        // find_halo_nodes<<<grid_size_G, BLOCK_SIZE>>>(d_row_ptr_G, d_col_idx_G, d_halo_nodes, induced_nodes_s, d_induced_nodes, d_pos);
        find_halo_nodes1<<<induced_nodes_s, BLOCK_SIZE>>>(d_row_ptr_G, d_col_idx_G, d_halo_nodes, induced_nodes_s, d_induced_nodes, d_pos);
        cudaDeviceSynchronize();
        gettimeofday(&end2, 0);
        long seconds = end2.tv_sec - begin2.tv_sec;
        long microseconds = end2.tv_usec - begin2.tv_usec;
        double elapsed2 = seconds + microseconds*1e-6;            
        cout<<"Time measured for Halo Nodes creation: "<<elapsed2<<" seconds."<<endl;
        cout << "Kernel code completed" << endl;
        // Retrieve halo nodes from device
        signed long long *h_halo_nodes = (signed long long *)malloc(num_edges * sizeof(signed long long));
        cudaMemcpy(h_halo_nodes, d_halo_nodes, num_edges * sizeof(signed long long), cudaMemcpyDeviceToHost);        
        unsigned long long *filtered_halo_nodes = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
        unsigned long long filtered_halo_nodes_size = 0;
        struct timeval begin3, end3;
        gettimeofday(&begin3, 0);
        for (unsigned long long i = 0; i < num_edges; ++i) {
            if (h_halo_nodes[i] != -1) {
                filtered_halo_nodes[filtered_halo_nodes_size++] = h_halo_nodes[i];
            }
        }
        // cout << "Pseudo nodes removed" << endl;
        cout << "Halo nodes (raw): ";
        for (unsigned long long i = 0; i<num_nodes; i++) {
            if (h_halo_nodes[i] != -1) {
                cout << h_halo_nodes[i] << " ";
            }
            else break;
        }
        cout << endl;

        // Remove duplicates
        sort(filtered_halo_nodes, filtered_halo_nodes + filtered_halo_nodes_size);
        auto end_unique = unique(filtered_halo_nodes, filtered_halo_nodes + filtered_halo_nodes_size);
        filtered_halo_nodes_size = distance(filtered_halo_nodes, end_unique);
        gettimeofday(&end3, 0);
        long seconds2 = end3.tv_sec - begin3.tv_sec;
        long microseconds2 = end3.tv_usec - begin3.tv_usec;
        double elapsed3 = seconds2 + microseconds2*1e-6;            
        cout<<"Time measured for duplication removal: "<<elapsed3<<" seconds."<<endl;

        // Combine induced subgraph and halo nodes
        unsigned long long h_combined_nodes_s = indices_size + filtered_halo_nodes_size;
        unsigned long long *h_combined_nodes = (unsigned long long *)malloc(h_combined_nodes_s * sizeof(unsigned long long));
        copy(indices, indices + indices_size, h_combined_nodes);
        copy(filtered_halo_nodes, filtered_halo_nodes + filtered_halo_nodes_size, h_combined_nodes + indices_size);
        outfile << h_combined_nodes_s<<endl;
        cout << "Number of Combined nodes: "<<h_combined_nodes_s<<endl;
        for (unsigned long long i = 0; i < h_combined_nodes_s; ++i) {
            outfile << h_combined_nodes[i] << " ";
            cout << h_combined_nodes[i] << " ";
        }
        cout << endl;
        outfile<<endl;


        // for (unsigned long long i = 0; i < h_combined_nodes_s; i++) {
        //     outfile << row_ptr_part[i] << " ";
        // }

        // Prepare result buffers

        free(indices);
        free(h_halo_nodes);         
        free(filtered_halo_nodes);         
        free(h_combined_nodes);

        cudaFree(d_induced_nodes);
        cudaFree(d_pos);
        cudaFree(d_halo_nodes);
        cudaFree(d_row_ptr_G);
        cudaFree(d_col_idx_G);
    }
   outfile.close();
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <nopart> <filename>" << endl;
        return 1;
    }

    // Parse command-line arguments
    unsigned long long nopart = stoi(argv[1]);
    string filename = argv[2];

    // Read the input file
    ifstream infile(filename);
    if (!infile.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return 1;
    }

    unsigned long long num_nodes, num_edges;
    infile >> num_nodes >> num_edges;

    unsigned long long num_master_nodes, row_ptr_size, col_idx_size, num_halo_nodes;
    infile >> num_master_nodes >> row_ptr_size >> col_idx_size >> num_halo_nodes;

    unsigned long long *row_ptr_G = (unsigned long long *)malloc(row_ptr_size * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < row_ptr_size; ++i) {
        infile >> row_ptr_G[i];
    }

    unsigned long long *col_idx_G = (unsigned long long *)malloc(col_idx_size * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < col_idx_size; ++i) {
        infile >> col_idx_G[i];
    }

    unsigned long long *in_deg = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < num_nodes; ++i) {
        infile >> in_deg[i];
    }
    // for(unsigned long long i = 0; i<num_nodes; i++) cout<<in_deg[i]<<" ";
    infile.close();

    string filename2 = argv[3];

    // Read the input file
    ifstream infile2(filename2);
    if (!infile2.is_open())
    {
        cerr << "Error opening file: " << filename2 << endl;
        return 1;
    }

    unsigned long long *node_parts = (unsigned long long *)malloc(num_nodes * sizeof(unsigned long long));
    for (unsigned long long i = 0; i < num_nodes; ++i) {
        infile2 >> node_parts[i];
    }

    infile2.close();

    string output_filename = filename + argv[1] + "_output.csr";
    // Call the function to create subgraph with halo nodes
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    create_subgraph_with_halo(nopart, node_parts, row_ptr_G, col_idx_G, in_deg, output_filename,num_nodes,num_edges);
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    cout<<"Time measured:"<<elapsed<<" seconds."<<endl;

    return 0;
}
