#!/usr/bin/env bash

read -p "Maxindex (1000*2^i): " maxindex
export PYTHONPATH=.

for ((i = $maxindex ; i >= 0; i--)); do
    for alpha in 0 0.25 0.5 0.75 1; do
    echo "Running: Alpha =" $alpha "n =" $i
    mpirun -np 1 python apps/convergence/script_convergence.py $alpha $i $maxindex&
    done
    wait
done

echo "All done!"