import cupy as cp
from cupy.cuda import function

# # CUDA kernel code for segmented radix sort
# KERNEL_CODE = '''
# extern "C" {
#     __global__ void segmented_radix_sort_kernel(const int* keys_in, int* keys_out, const int num_items,
#                                                 const int num_segments, const int* offsets) {
#         int tid = blockDim.x * blockIdx.x + threadIdx.x;
#         if (tid < num_items) {
#             int segment_id = 0;
#             for (int i = 0; i < num_segments; ++i) {
#                 if (tid >= offsets[i] && tid < offsets[i + 1]) {
#                     segment_id = i;
#                     break;
#                 }
#             }
#
#             int key = keys_in[tid];
#             int bit_mask = 1;
#             for (int bit = 0; bit < 32; ++bit) {  // Assuming 32-bit integers
#                 __shared__ int count[32];
#                 count[threadIdx.x] = 0;
#                 __syncthreads();
#
#                 for (int i = blockDim.x * blockIdx.x; i < num_items; i += blockDim.x * gridDim.x) {
#                     if (segment_id == (i - i % blockDim.x + tid) / blockDim.x) {
#                         if (keys_in[i] & bit_mask) {
#                             atomicAdd(&count[bit], 1);
#                         }
#                     }
#                 }
#                 __syncthreads();
#
#                 int offset = 0;
#                 for (int i = 0; i < threadIdx.x; ++i) {
#                     offset += count[i];
#                 }
#                 keys_out[offset] = key;
#                 bit_mask <<= 1;
#                 __syncthreads();
#             }
#         }
#     }
# }
# '''
# CUDA kernel code for segmented radix sort
KERNEL_CODE = '''
extern "C" {
    __global__ void segmented_radix_sort_kernel(const int* keys_in, int* keys_out, const int num_items,
                                                const int num_segments, const int* offsets) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < num_items) {
            int segment_id = 0;
            for (int i = 0; i < num_segments; ++i) {
                if (tid >= offsets[i] && tid < offsets[i + 1]) {
                    segment_id = i;
                    break;
                }
            }

            // Simple swap implementation (modify based on the actual sorting logic)
            int temp;
            for (int i = tid + 1; i < num_items; ++i) {
                if (segment_id == (i - i % blockDim.x + threadIdx.x) / blockDim.x) {
                    if (keys_in[tid] > keys_in[i]) {
                        temp = keys_in[tid];
                        keys_in[tid] = keys_in[i];
                        keys_in[i] = temp;
                    }
                }
            }
            keys_out[tid] = keys_in[tid];
        }
    }
}
'''
# Compile the CUDA kernel code
#segmented_radix_sort_kernel = function.Module().compile(KERNEL_CODE).get_function('segmented_radix_sort_kernel')
segmented_radix_sort_kernel = cp.RawKernel(KERNEL_CODE, 'segmented_radix_sort_kernel')

# Function to perform segmented radix sort on GPU
def segmented_radix_sort(keys_in, num_segments, offsets):
    num_items = len(keys_in)
    keys_out = cp.zeros_like(keys_in)

    threads_per_block = 128
    blocks_per_grid = (num_items + threads_per_block - 1) // threads_per_block

    segmented_radix_sort_kernel((blocks_per_grid,), (threads_per_block,),
                                (keys_in, keys_out, num_items, num_segments, offsets))

    return keys_out

# Example usage
num_segments = 3
offsets = cp.array([0, 3, 3, 7], dtype=cp.int32)
keys_in = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=cp.int32)

sorted_keys = segmented_radix_sort(keys_in, num_segments, offsets)
print("Sorted Keys:", sorted_keys)
