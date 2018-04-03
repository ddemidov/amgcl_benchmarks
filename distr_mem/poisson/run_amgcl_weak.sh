#!/bin/bash

echo --- Weak ---
for nc in 96 384 1536 6144; do
    echo --- Cores: ${nc} ---
    for omp in 1 12; do
        echo --- OpenMP: ${omp} ---
        echo --- Linear ---
        ./run_amgcl.py --np $((${nc} / ${omp})) --omp ${omp}
        echo --- Constant ---
        ./run_amgcl.py --np $((${nc} / ${omp})) --omp ${omp} --const
    done
done
