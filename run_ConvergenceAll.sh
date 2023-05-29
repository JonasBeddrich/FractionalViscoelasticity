#!/usr/bin/env bash

read -p "Maximum numer of steps per second (1000*2^i): " maxindex
export PYTHONPATH=.

for ((i = $maxindex ; i >= 0; i--)); do
    for alpha in 0 0.25 0.5 0.75 1; do
    echo "Running: Alpha =" $alpha "n =" $i
    mpirun -np 1 python script_Convergence.py config_Convergence $1 $alpha $i $maxindex&
    done
    wait
done

echo "All done!"