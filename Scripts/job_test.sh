#!/bin/bash

# Set / as the delimiter
IFS='/'

# Split input string into an array of words
read -ra words <<< "$1"

# Echo last word of the array
filename="${words[-1]}"
prefix="output_"
echo $prefix$filename
