#!/usr/bin/env python3
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--np',  dest='np', required=True, type=int)
parser.add_argument('--dpp', dest='dpp', default=100**3, type=int)
args = parser.parse_args(sys.argv[1:])

script_name = f'scripts/petsc_{args.np}_{args.dpp}'
params      = f'-n {int(0.5 + (args.dpp * args.np)**(1/3))}'

script_name += '.sbatch'

open(script_name, 'w').write(
f"""#!/bin/bash
#SBATCH --job-name=petsc
#SBATCH -D .
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={args.np}
#SBATCH --time=00:10:00

export OMP_NUM_THREADS=1

echo "{params}"
srun ./dmem_poisson_petsc {params}
""")

subprocess.Popen(['sbatch', script_name]).wait()
