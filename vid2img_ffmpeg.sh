#!/bin/bash
#1-filename
#2-period
#3-path to save


if [ -n "$3" ]
then
dirname=$3
else
filename=$(basename $1)
dirname=$(echo $filename | cut -d'.' -f1)
fi

if [ -n "$2" ]
then
period=$2
else
period=1
fi

if [ -n "$1" ]
then
ffmpeg -i $1 -r $period $dirname"_"%07d.jpg
echo create dir $dirname 
mkdir $dirname
mv $dirname*.jpg $dirname 
else
echo "file not set"
fi







