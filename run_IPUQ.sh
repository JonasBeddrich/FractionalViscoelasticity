#!/usr/bin/env bash

read -p "Number of runs: " n
read -p "Starting alpha: " alpha0
read -p "Noise perc:     " noise
read -p "Folder name:    " folder

echo "Creating data"
for i in $(seq -f "%013g" 1 $n) 
do
  taskset -c $i python script_GenerateData.py config_IPStability $i &
done
wait

echo "Inverse Problem"
for i in $(seq -f "%013g" 1 $n) 
do
  taskset -c $i python script_IPStability.py config_IPStability $i $alpha0 $noise $folder &
done
wait

echo "Make Plot"
mpirun -np 1 python script_MakePlots_IPUQ.py config_IPStability $folder

echo "All done!"