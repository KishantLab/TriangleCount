path=$(cd .. && pwd)
tsv_path="/Dataset/tsv_files"

for file in $path$tsv_path/*_adj.tsv
do
	trimmed=$(basename $file _adj.tsv)
#line="$(fgrep -w $trimmed $path/Dataset/snap_metadata.csv)"
#IFS=','
#read -a strarr <<<"$line"
#vertex=${strarr[3]}

./convert_deg.sh $file
done
