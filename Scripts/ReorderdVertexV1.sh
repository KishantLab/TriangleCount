#!/bin/bash
vertex=0
path=$(cd .. && pwd)
#echo $path
#echo "File_Name, Vertex, Edge, Triangle, Result, Diffrence, Status, time(Kernel) "
str2="/Dataset/Undirected/"
str3="/Dataset/Directed/"
reorder_dir="/Dataset/Graphs/"
dir="deg_dir.txt"
part="_adj.graph.part.$2"
file=$path$str3$1
#for file in $path$str2*.txt
#do
 echo $file
 trimmed=$(basename $1 deg_dir.txt)
 #echo "$trimmed"
line="$(fgrep -w $trimmed $path/Dataset/snap_metadata.csv)"
IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
triangle=${strarr[2]}
edges=${strarr[1]}
reorder_file=$path$reorder_dir$trimmed$part
#echo $reorder_file
#echo $file "         " "Total Vertex :" $vertex
cd ..
cd src
#echo $trimmed$str1   $file
#nvcc TC_v1.cu
nvcc ReorderdVertexV1.cu 
./a.out $file $vertex $reorder_file $2
#done
