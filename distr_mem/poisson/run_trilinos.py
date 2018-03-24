#!/usr/bin/env python3
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--np',  dest='np', required=True, type=int)
parser.add_argument('--dpp', dest='dpp', default=100**3, type=int)
parser.add_argument('--rebalance', dest='rebalance')
parser.add_argument('--dd_ml', dest='dd_ml', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

script_name = f'trilinos_{args.np}_{args.dpp}'
params      = f'--n={int(0.5 + (args.dpp * args.np)**(1/3))}'

if args.rebalance:
    script_name += f'_{args.rebalance.lower()}'
    params      += f' --r={args.rebalance}'

if args.dd_ml:
    script_name += '_ddml'
    params      += ' --dd=1'

script_name += '.sbatch'

open(script_name, 'w').write(
f"""#!/bin/bash
#SBATCH --job-name=trilinos
#SBATCH -D .
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={args.np}
#SBATCH --time=00:10:00

export OMP_NUM_THREADS=1

srun ./dmem_poisson_trilinos {params}
""")

subprocess.Popen(['sbatch', script_name]).wait()
