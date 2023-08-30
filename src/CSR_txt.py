'''
import sys
import csv
from numpy import array
from scipy.sparse import coo_matrix
row =[]
col =[]
data =[]
v=sys.argv[3]
vertex = int(v)+1
file = open(sys.argv[1],'r')

tsv_file = csv.reader(file, delimiter="\t")

for line in tsv_file:
	row.append(int(line[0]))
	col.append(int(line[1]))
	data.append(int(line[2]))

#print(row)
#print(col)
#print(data)
A = coo_matrix((data, (row, col))).tocsr()
'''
from numpy import array
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import csv
import sys
v=sys.argv[3]
vertex = int(v)+1
file=pd.read_csv(sys.argv[1],delimiter=' ')
print(file['Dest'])
dest=file['Dest']

dest=np.array(dest)
file['Data']=1
data=file['Data']
data=np.array(data)
source=file['Source']
source=np.array(source)
A=coo_matrix((data, (source, dest))).tocsr()
nnz=A.getnnz()

file = open(sys.argv[2],'w')

file.write(str(A.nnz) + "\n")
#print("len of : ",len(A.indptr))
for data in range(vertex+1) :
	if (data < len(A.indptr)) :
		file.write("%i " % A.indptr[data])
	else:
		file.write("%i " % A.nnz)
file.write("\n")
for data in A.indices :
	file.write("%i " % data)
print(A.nnz)
print(A.indptr)
print(A.indices)

