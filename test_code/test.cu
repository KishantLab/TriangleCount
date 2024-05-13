#include <iostream>
#include <cuda_runtime.h>

const int N = 4;  // Number of vertices in the original graph

// Helper function to print CSR format
void printCSR(int* values, int* row_ptr, int* col_indices, int num_rows) {
    std::cout << "Values: ";
    for (int i = 0; i < row_ptr[num_rows]; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Row Pointer: ";
    for (int i = 0; i <= num_rows; ++i) {
        std::cout << row_ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Column Indices: ";
    for (int i = 0; i < row_ptr[num_rows]; ++i) {
        std::cout << col_indices[i] << " ";
    }
    std::cout << std::endl;
}

// CUDA kernel to construct induced subgraph using CSR format
__global__ void constructSubgraphCSR(int* originalValues, int* originalRowPtr, int* originalColIndices,
                                      int* subgraphValues, int* subgraphRowPtr, int* subgraphColIndices,
                                      int* vertices, int numVertices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numVertices) {
        int subgraphIdx = vertices[tid];
        subgraphRowPtr[tid + 1] = originalRowPtr[subgraphIdx + 1] - originalRowPtr[subgraphIdx];

        for (int i = originalRowPtr[subgraphIdx]; i < originalRowPtr[subgraphIdx + 1]; ++i) {
            int originalIdx = originalColIndices[i];
            subgraphColIndices[i - originalRowPtr[subgraphIdx]] = originalIdx;
            subgraphValues[i - originalRowPtr[subgraphIdx]] = originalValues[i];
        }
    }
}

int main() {
    // Define the original graph in CSR format
    int originalValues[] = {1, 1, 1, 1, 1, 1, 1};
    int originalRowPtr[] = {0, 2, 4, 6, 7};
    int originalColIndices[] = {1, 2, 0, 2, 0, 1, 3};

    // Define the vertices of the induced subgraph
    int subgraphVertices[] = {1, 2, 3};  // Example: vertices 1, 2, and 3

    // Allocate device memory
    int* d_originalValues, * d_originalRowPtr, * d_originalColIndices;
    int* d_subgraphValues, * d_subgraphRowPtr, * d_subgraphColIndices;
    int* d_vertices;

    cudaMalloc((void**)&d_originalValues, sizeof(originalValues));
    cudaMalloc((void**)&d_originalRowPtr, sizeof(originalRowPtr));
    cudaMalloc((void**)&d_originalColIndices, sizeof(originalColIndices));

    cudaMalloc((void**)&d_subgraphValues, sizeof(originalValues));
    cudaMalloc((void**)&d_subgraphRowPtr, sizeof(subgraphVertices) + 1);
    cudaMalloc((void**)&d_subgraphColIndices, sizeof(originalColIndices));

    cudaMalloc((void**)&d_vertices, sizeof(subgraphVertices));

    // Copy data to device
    cudaMemcpy(d_originalValues, originalValues, sizeof(originalValues), cudaMemcpyHostToDevice);
    cudaMemcpy(d_originalRowPtr, originalRowPtr, sizeof(originalRowPtr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_originalColIndices, originalColIndices, sizeof(originalColIndices), cudaMemcpyHostToDevice);

    cudaMemcpy(d_vertices, subgraphVertices, sizeof(subgraphVertices), cudaMemcpyHostToDevice);

    // Launch the kernel
    int numBlocks = 1;
    int threadsPerBlock = N;
    constructSubgraphCSR<<<numBlocks, threadsPerBlock>>>(d_originalValues, d_originalRowPtr, d_originalColIndices,
                                                          d_subgraphValues, d_subgraphRowPtr, d_subgraphColIndices,
                                                          d_vertices, sizeof(subgraphVertices));

    // Copy result back to host
    int subgraphValues[N];
    int subgraphRowPtr[N + 1];
    int subgraphColIndices[N];

    cudaMemcpy(subgraphValues, d_subgraphValues, sizeof(subgraphValues), cudaMemcpyDeviceToHost);
    cudaMemcpy(subgraphRowPtr, d_subgraphRowPtr, sizeof(subgraphRowPtr), cudaMemcpyDeviceToHost);
    cudaMemcpy(subgraphColIndices, d_subgraphColIndices, sizeof(subgraphColIndices), cudaMemcpyDeviceToHost);

    // Print the induced subgraph in CSR format
    std::cout << "Original Graph (CSR format):" << std::endl;
    printCSR(originalValues, originalRowPtr, originalColIndices, N);

    std::cout << "\nInduced Subgraph (CSR format):" << std::endl;
    printCSR(subgraphValues, subgraphRowPtr, subgraphColIndices, sizeof(subgraphVertices));

    // Free device memory
    cudaFree(d_originalValues);
    cudaFree(d_originalRowPtr);
    cudaFree(d_originalColIndices);
    cudaFree(d_subgraphValues);
    cudaFree(d_subgraphRowPtr);
    cudaFree(d_subgraphColIndices);
    cudaFree(d_vertices);

    return 0;
}

