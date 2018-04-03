#!/bin/bash

echo --- Strong ---
for nc in 96 384 1536 6144; do
    echo --- Cores: ${nc} ---
    for omp in 1 12; do
        echo --- OpenMP: ${omp} ---
        echo --- Linear ---
        ./run_amgcl.py --np $((${nc} / ${omp})) --omp ${omp} --dpp 512 --strong
        echo --- Constant ---
        ./run_amgcl.py --np $((${nc} / ${omp})) --omp ${omp} --dpp 512 --strong --const
    done
done
