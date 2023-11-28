#!/usr/bin/env bash

read -p "Alpha: " alpha
read -p "Maximum numer of steps per second (1000*2^i): " maxindex
export PYTHONPATH=.

for ((i = 0 ; i <= $maxindex; i++)); do
    echo "Running n =" $i
    mpirun -np 1 python3 script_Convergence.py config_Convergence $1 $alpha $i $maxindex&
done

wait
echo "All done!"