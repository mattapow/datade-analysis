#!/bin/bash
# run a single liggghts simulation
# root path as first variable
# slope angle as second input

path_root=$1
mkdir $path_root
mkdir $path_root/CG
mkdir $path_root/dump
mkdir $path_root/results
mkdir $path_root/samples

# modify liggghts input file for each slope angle
sed "s/variable Angle equal 25/variable Angle equal $2/" in.liggghts > $path_root/in.liggghts

cd $path_root
lmp_serial -in in.liggghts
cd ..
