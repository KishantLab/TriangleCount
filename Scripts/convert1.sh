#!/bin/bash
vertex=0
if [ "$#" -eq 0 ]
then
    echo " 
    	Welcome to KishantLab Triangle Counting Algorithms 
    	To convert TSV file to another formate files just run  
	
	usage : 
		 ./convert.sh <file_name_adj.tsv> 
	
	file_name : TSV file which is dwonload from graph challange website and must src dst list .
        
	" 	
    exit 1
fi
path=$(cd .. && pwd)
#echo $path
#str1="_adj.graph.part.$2"
file=$1
#str1="_adj.graph.part.$2"
str2="/Dataset/Directed/"
str5="/Dataset/Undirected/"
str3="/Dataset/Graphs/"
str4="/Dataset/tsv_files/"
dir="_dir.txt"
adj="_adj.graph"
tsv="_adj.tsv"
txt="_adj.txt"
#for file in *.txt
#do
 #echo $file
 trimmed=$(basename $file _adj.tsv)
 #echo "$trimmed"
line="$(fgrep -w $trimmed $path/Dataset/snap_metadata.csv)"
IFS=','
read -a strarr <<<"$line"
vertex=${strarr[3]}
#triangle=${strarr[2]}
#edges=${strarr[1]}
graph_file=$trimmed$adj
Graph_file=$path$str3$graph_file
#echo $Graph_file 
part_file=$path$str3$trimmed$str1
dirfile=$path$str2$trimmed$dir
tsvfile=$path$str4$trimmed$tsv
txtfile=$path$str5$trimmed$txt
#echo $txtfile
#echo $file "         " "Total Vertex :" $vertex
cd ..
cd src
echo " CSR Formate Create ************************************"
python3 CSR_txt.py $tsvfile $txtfile $vertex
echo "
GRAPH File Create ********************************************"
gcc part_csr.c
./a.out $txtfile $vertex $Graph_file
echo "
Convert Undirected --> Directed ******************************"
gcc ConvertUndirToDir.c
./a.out $txtfile $dirfile $vertex
echo "
Files Creation Sucessfull ***********************************

Note : you should passed the number of partitions you want
now you run 

./auto_v4.sh $trimmed$dir <number> 

"
#done
