import cupy as cp

# Define a raw CUDA kernel
delete_duplicates_kernel = cp.RawKernel(r'''
extern "C" __global__
void delete_duplicates(int* input, int* output, int* output_len, int* seen, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < N) {
        int val = input[idx];
        
        // Use atomic operations to ensure only one thread processes a unique value
        int old = atomicCAS(&seen[val], 0, 1);
        
        if (old == 0) {
            int pos = atomicAdd(output_len, 1);
            output[pos] = val;
        }
    }
}
''', 'delete_duplicates')

def delete_duplicates(input_list):
    input_array = cp.array(input_list, dtype=cp.int32)
    N = input_array.size
    
    output_array = cp.zeros(N, dtype=cp.int32)
    output_len = cp.zeros(1, dtype=cp.int32)
    
    # Create a seen array with the same range as the possible values in the input
    max_val = cp.max(input_array).item()
    seen = cp.zeros(max_val + 1, dtype=cp.int32)
    
    # Calculate grid and block dimensions
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
    
    # Launch the kernel
    delete_duplicates_kernel(
        (blocks,), (threads_per_block,),
        (input_array, output_array, output_len, seen, N)
    )
    
    # Copy output length and output array to host
    output_len_host = output_len.get().item()  # Get the scalar value from the array
    output_array_host = output_array[:output_len_host].get()
    
    return output_array_host.tolist()

# Example usage
input_list = [1, 7, 3, 2, 7, 4, 9, 3]
output_list = delete_duplicates(input_list)
print("Input List:", input_list)
print("Output List without duplicates:", output_list)

