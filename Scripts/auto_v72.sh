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
str1="_adj.graph.part.$2"
new_csr="_adj.graph.part.csr.$2"
file=$1
str2="/Dataset/Directed/"
str3="/Dataset/Graphs/"
str4="/Dataset/input_csr/"
dir="_dir.txt"
adj="_adj.graph"
trimmed=$(basename $file _dir.txt)
line="$(fgrep -w $trimmed $path/Dataset/snap_metadata.csv)"
IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
edges=${strarr[1]}
Graph_file=$path$str3$trimmed$str1

if [ -f "$path$str3$trimmed$str1" ]; then
	echo "********partition $trimmed$str1 exist************"
else
	gpmetis -ptype=kway -ctype=shem -objtype=cut -contig -minconn $Graph_file $2
fi

#gpmetis -ptype=kway -ctype=shem -objtype=cut -contig -minconn $Graph_file $2

part_file=$path$str3$trimmed$str1
file_name=$path$str2$file
new_csr_file=$path$str4$trimmed$new_csr
cd ..
cd src
#nvcc TC_v41.cu
if [ -f "$new_csr_file" ]; then
	echo "*******preprocessed file $trimmed$new_csr exist******"
else
	gcc preprocess.c -o preprocess
	./preprocess $part_file $file_name $vertex $2 $new_csr_file
fi
nvcc TC_v72.cu
./a.out $new_csr_file $vertex $edges $2
#gcc part_array_v1.c
#./a.out $part_file $file $vertex $1
#done
