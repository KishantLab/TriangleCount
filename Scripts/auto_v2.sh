#! /bin/bash
vertex=0
if [ "$#" -eq 0 ]
then
    echo " 
    	Welcome to KishantLab Triangle Counting Algorithms 
    	 
	usage : 
		 ./auto_v2.sh <file_name.dir> part 
	
	file_name : file name where stored the directed graph information (CSR format) .
        
	part : pass the total  number of partitions you want to do that in a computation .
	
	" 	
    exit 1
fi
#str1="_adj.graph.part.$2"
file=$1
str2="/Dataset/Directed/"
str3="/Dataset/Undirected/"
str4="/Dataset/Graphs/"
dir="_dir.txt"
adj="_adj.graph"
#triangle=0
#echo "File_Name, Vertex, Edge, Triangle, partition(1), partition(2), partition(3), total__time "
#for file in *.txt
#do
 #echo $file
 trimmed=$(basename $file _dir.txt)
 #echo "$trimmed"
line="$(fgrep -w $trimmed snap_metadata.csv)"
IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
#triangle=${strarr[2]}
#edges=${strarr[1]}
graph_file=$trimmed$adj
Graph_file=$str4$graph_file
gpmetis -ptype=kway -ctype=shem -objtype=cut -contig -minconn $Graph_file $2

str1="_adj.graph.part.$2"

part_file=$str3$trimmed$str1
file_name=$str2$file
#echo $file "         " "Total Vertex :" $vertex
cd ..
cd src
nvcc TC_v2.cu
./a.out $part_file $file_name $vertex $2 
#gcc part_array_v1.c
#./a.out $part_file $file $vertex $1
#done
