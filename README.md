# TriangleCount
LSTC is a project that provides a high-performance and Memory Efficient Graph Analytical Framework for GPUs. We use Cuda and various graph partitioning-based parallel programs for counting the number of triangles in a CSR sparse matrix graphs.
# Requirements 
DGL is a framework that provides the program's graph transformations.
* Download, build and install Nvidia CUDA.
   - Note: We use Cuda 11.7 for this project.
* Download, build, and install  DGL 2.x , on [link](https://docs.dgl.ai/en/2.0.x/install/) Document.
  ```
  pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
  ```
  - Note: please see your compatible Cuda version and install dgl based on this.
---

## How to Run

### 1. Download the Dataset

Download the dataset from websites like [GraphChallenge](https://graphchallenge.mit.edu) or [SuiteSparse](https://sparse.tamu.edu).

### 2. Clean the Dataset and Convert to Undirected CSR

1. Navigate to the preprocessing directory:
    ```
    cd preprocess_GPU/generate_clean_undir_dataset/
    ```
2. Run the preprocessing script:
    ```
    python3 Halo_creation_preprocess.py <dataset_file>
    ```
   Example:
    ```
    python3 Halo_creation_preprocess.py ../data/Dataset/GraphChallange/amazon0302_adj.tsv
    ```
3. This generates the undirected CSR file `<filename.csr>` and partitioned node parts files `<filename.csr_node_parts_<part_method>_<part_number>`.
4. Create the induced subgraph by cleaning the redundant vertex and edges:
    ```
    cd ../../
    cd preprocess_GPU/Halo_cleanned_preprocess/
    ./round_clean_sg <part_number> <filename.csr> <filename.csr_node_parts_<part_method>_<part_number>
    ```
    Example:
    ```
    ./round_clean_sg 2 it-2004_convert_part.csr it-2004_convert_part.csr_node_parts_random_2
    ```
5. The generated file is a cleaned induced subgraph for triangle counting.

### 3. Triangle Count with Partition

1. Navigate to the source directory:
    ```
    cd TriangleCount/src/WithPartition/
    ```
2. Compile the code:
    ```
    ./compile.sh
    ```
3. Run the triangle count:
    ```
    ./wo_sort_tc_256 <filename> <part_number>
    ```
    Example:
    ```
    ./wo_sort_tc_256 ../LSTC/Dataset_part/it-2004_convert_part.csr_node_parts_random_2_output.csr 2
    ```
### 4. Triangle Count Without Partition

1. Preprocess the data for triangle counting without partition:
    ```sh
    cd TriangleCount/Med_dataset/
    python3 without_part_preprocess.py <filename>
    ```
2. Navigate to the source directory:
    ```sh
    cd ../src/WithoutPartition/
    ```
3. Compile the code:
    ```sh
    ./compile.sh
    ```
4. Run the triangle count:
    ```sh
    ./TC_128 <filename.csr>
    ```
   Example:
    ```sh
    ./TC_128 /data/kishan/TriangleCount/Med_dataset/wikipedia/wikipedia_link_en_output.csr
    ```

### 5. Compare Different Parameters

1. Navigate to the source directory:
    ```sh
    cd src/WithoutPartition/
    ```
2. Run with atomic operations:
    ```sh
    ./With_Atomic_op <filename.csr>
    ```
3. Run without shared memory operations:
    ```sh
    ./Without_ShareMemory_TC <filename.csr>
    ```

### 6. Profiling Options

1. Navigate to the profiling directory:
    ```sh
    cd src/profiling/
    ```
2. Compile the code:
    ```sh
    ./compile.sh
    ```
3. Run the profiling script:
    ```sh
    ./run.sh
    ```
    Note: In `run.sh`, pass the `<filename.csr>` generated in `Med_dataset/without_part_preprocess.py` file.

## Authors

- Kishan Tamboli
- Vinayak Kesarwani
