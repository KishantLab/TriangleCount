nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o ncu_profiling ShareKernelTCV62_update_v1.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o without_shared_mem Without_ShareMemory_TC.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o with_Atomic With_Atomic_op.cu
#With_Atomic_op.cu
#iWithout_ShareMemory_TC.cu
