#! /bin/bash
vertex=0
path=$(cd .. && pwd)
echo $path
#str1="_adj.graph.part.$2"
file=$1
str2="/Dataset/Directed/"
str3="/Dataset/Graphs/"
dir="_dir.txt"
adj="_adj.graph"
cd ..
cd Dataset/Directed/
for file in *.txt
do
 #echo $file
 trimmed=$(basename $file _dir.txt)
 #echo "$trimmed"
line="$(fgrep -w $trimmed $path/Dataset/snap_metadata.csv)"
IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
#triangle=${strarr[2]}
#edges=${strarr[1]}
graph_file=$trimmed$adj
Graph_file=$path$str3$graph_file
gpmetis -ptype=kway -ctype=shem -objtype=cut -contig -minconn $Graph_file 16
str1="_adj.graph.part.16"
part_file=$path$str3$trimmed$str1
file_name=$path$str2$file
#echo $file "         " "Total Vertex :" $vertex
cd
cd $path/src
echo $(pwd)
echo $trimmed$str1   $file
nvcc TC_v3.cu
./a.out $part_file $file_name $vertex 16
#gcc part_array_v1.c
#./a.out $part_file $file $vertex $1
done
