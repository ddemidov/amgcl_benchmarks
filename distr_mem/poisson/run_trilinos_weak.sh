#!/bin/bash

echo --- Weak ---
for nc in 96 384 1536 6144; do
    echo --- Cores: ${nc} ---
    echo --- ML ---
    ./run_trilinos.py --np ${nc} --rebalance Zoltan
    echo --- DD/ML ---
    ./run_trilinos.py --np ${nc} --rebalance Zoltan --dd_ml 2
done
