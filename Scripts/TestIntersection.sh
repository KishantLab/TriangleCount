#! /bin/bash
vertex=0
path=$(cd .. && pwd)
#echo $path
echo "File_Name, Vertex, Edge, Triangle, Result, Diffrence, Status, time(Kernel) "
str2="/Dataset/Undirected/"
str3="/Dataset/Directed/"
dir="_adj.txt"
file=$path$str2$1
#for file in $path$str2*.txt
#do
 #echo $file
 trimmed=$(basename $file _adj.txt)
 #echo "$trimmed"
line="$(fgrep -w $trimmed $path/Dataset/snap_metadata.csv)"
IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
triangle=${strarr[2]}
edges=${strarr[1]}
#echo $file "         " "Total Vertex :" $vertex
cd ..
cd src
#echo $trimmed$str1   $file
#nvcc TC_v1.cu
nvcc TestIntersection.cu 
./a.out $file $vertex $triangle $trimmed 
#done
