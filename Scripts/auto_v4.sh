#!/bin/bash
vertex=0
if [ "$#" -eq 0 ]
then
    echo " 
        Welcome to KishantLab Triangle Counting Algorithms(TCSG)  
    	 
	usage : 
		 ./auto_v2.sh <file_name_dir.txt> part 
	
	file_name : file name where stored the directed graph information (CSR format) .
        
	part : pass the total  number of partitions you want to do that in a computation .
	
	" 	
    exit 1
fi
path=$(cd .. && pwd)
#echo $path
str1="_adj.graph.part.$2"
file=$1
str2="/Dataset/Directed/"
str3="/Dataset/Graphs/"
dir="_dir.txt"
adj="_adj.graph"
#for file in $path$str2*.txt
#do
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

File=$Graph_file
if [ -f "$path$str3$trimmed$str1" ]; then
echo "********partition $graph_file$str1 exist************"
else
gpmetis -ptype=kway -ctype=shem -objtype=cut -contig -minconn $Graph_file $2
fi
#gpmetis -ptype=kway -ctype=shem -objtype=cut -contig -minconn $Graph_file $2

#str1="_adj.graph.part.3"

part_file=$path$str3$trimmed$str1
file_name=$path$str2$file
#echo $file "         " "Total Vertex :" $vertex
cd ..
cd src
echo $trimmed$str1   $file
nvcc TC_v3.cu
./a.out $part_file $file_name $vertex $2 
#gcc part_array_v1.c
#./a.out $part_file $file $vertex $1
#done
