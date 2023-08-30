# TriangleCount
Triangle counting is a project which is provide A Hiqh Performance and Memory Efficient Graph Analytical Framework for GPUs”**(2008200)
a cuda and metis (graph partitioning) based parallel program for counting the number of triangle in a CSR sparce matrix graphs .
# Requirements 
METIS is framework that provide a programs for partitioning graphs .
* Download , build, and install  Metis 5.0 , on [METIS](https://github.com/KarypisLab/METIS) Github .
* Download , build and install Nvidia CUDA .
# Versions
* **ShareMemory Executions** 
- **_ShareKernelTCV62.cu_** is a stable version of Exact Triangle Counting.
- note : Before Execution make sure N_THREADS_PER_BLOCK 256 and SHARED_MEM 256 . 
- That is a best configuration for better performance.

- **Follow the Instructions.**
```
cd TriangleCount/Scripts
```
- **Example **
```
 ./ShareKernelTCV62.sh <FILENAME>
 ./ShareKernelTCV62.sh amazon0601deg_dir.txt
 ```
- **Graph500 Files***
```
graph500-scale18-ef16deg_dir.txt
graph500-scale19-ef16deg_dir.txt
graph500-scale20-ef16deg_dir.txt
graph500-scale21-ef16deg_dir.txt
graph500-scale22-ef16deg_dir.txt
cit-Patentsdeg_dir.txt
amazon0601deg_dir.txt
```
- ** Dataset Path**
```
/home/kishan/TriangleCount/Dataset/Directed
```
