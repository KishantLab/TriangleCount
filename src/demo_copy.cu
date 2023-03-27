#include <cuda.h>
#include <stdio.h>

#define N 2

__global__ void K(int *out, int *in, int size) {
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        out[id] = in[id] * in[id];
}
int main() {
        cudaStream_t stream[N];
        for (unsigned ii = 0; ii < N; ++ii)
                cudaStreamCreate(&stream[ii]);

        int *hptr, *dinptr, *doutptr;
        unsigned nbytesperstream = (1<<10);
        printf("nbytesperstream : %d",nbytesperstream);
        unsigned nbytes = N * nbytesperstream;
        printf("nbytes : %d",nbytes);
        cudaHostAlloc(&hptr, nbytes, 0);
        cudaMalloc(&dinptr, nbytes);
        cudaMalloc(&doutptr, nbytes);

        for (unsigned ii = 0; ii < N; ++ii) {
                cudaMemcpyAsync(dinptr + ii * nbytesperstream, hptr + ii * nbytesperstream, nbytesperstream, cudaMemcpyHostToDevice, stream[ii]);
                K<<<nbytesperstream / 512, 512, 0, stream[ii]>>>(doutptr + ii * nbytesperstream, dinptr + ii * nbytesperstream, nbytesperstream);
                cudaMemcpyAsync(hptr + ii * nbytesperstream, doutptr + ii * nbytesperstream, nbytesperstream, cudaMemcpyDeviceToHost, stream[ii]);
        }

        return 0;
}
