path=$(cd .. && pwd)
Dir="/Dataset/Directed/"
str="_dir.txt"

for file in $path$Dir*_dir.txt
do
  trimmed=$(basename $file _dir.txt)
  in=$trimmed$str
  ./DglPreprocess.sh $in 3
  ./DglShareKernelV1.sh $in 3
done
