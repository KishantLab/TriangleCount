nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o updated_tc Final_updated_corrected.cu
nvcc --gpu-architecture=compute_86 --gpu-code=sm_86 -o wo_sort_tc_256 without_sorting_tc.cu
