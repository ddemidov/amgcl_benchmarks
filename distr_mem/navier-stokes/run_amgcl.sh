#!/bin/bash

echo --- AMGCL_NS ---
for nc in 96 384 1536 6144; do
    echo --- Cores: ${nc} ---
    for omp in 1 12; do
        echo --- OpenMP: ${omp} ---
        ./run_amgcl.py --np $((${nc} / ${omp})) --omp ${omp}
    done
done
