nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o TC_128 ShareKernelTCV62_update_v1.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o TC_256 Triangle_count_final.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o unified_Mem_TC Unified_memory_TC.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o With_Atomic_op With_Atomic_op.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o Without_ShareMemory_TC Without_ShareMemory_TC.cu

