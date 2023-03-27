#!/bin/bash
#vertex=0
path=$(cd .. && pwd)
new_csr=".part.csr.$2"
file=$1
str2="/Dataset/Directed/"
#str3="/Dataset/MetisPart/"
str4="/Dataset/input_csr/"
dir="deg_dir.txt"
#json=".json"
trimmed=$(basename $file deg_dir.txt)
echo $trimmed
#IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
edges=${strarr[1]}

#in_file=$path$str2$1
in_file=$path$str4$trimmed$new_csr

#part_file=$path$str3$trimmed$str1
#file_name=$path$str2$file
#new_csr_file=$path$str4$trimmed$new_csr
cd ..
cd src
#python3 SGpreprocess.py $in_file $out_file $2
nvcc ShareKernel2.cu
./a.out $in_file $2
#nvcc TC_v41.cu
#nvcc PShareKernelTCV1.cu
#./a.out $new_csr_file $vertex $edges $2
#gcc part_array_v1.c
#./a.out $part_file $file $vertex $1
#done
