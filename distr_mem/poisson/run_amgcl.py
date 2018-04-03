#!/usr/bin/env python3
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--np',  dest='np', required=True, type=int)
parser.add_argument('--dpp', dest='dpp', default=100**3, type=int)
parser.add_argument('--omp', dest='omp', default=1, type=int)
parser.add_argument('--strong', dest='strong', action='store_true', default=False)
parser.add_argument('--const', dest='const', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

script_name = f'scripts/amgcl_{args.np}x{args.omp}_{args.dpp}'
params = f'-p isolver.maxiter=500 local.coarse_enough=500 local.coarsening.aggr.eps_strong=0.0 -i cg -d pastix'

if args.strong:
    params += f' -n {args.dpp}'
else:
    params += f' -n {int(0.5 + (args.np * args.omp * args.dpp)**(1/3))}'

if args.const:
    script_name += '_const'
    params += ' --cd'

script_name += '.sbatch'

open(script_name, 'w').write(
f"""#!/bin/bash
#SBATCH --job-name=amgcl
#SBATCH -D .
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task={args.omp}
#SBATCH --ntasks={args.np}
#SBATCH --time=00:10:00

export OMP_NUM_THREADS={args.omp}
export OMP_WAIT_POLICY=ACTIVE

if [ {args.omp} -gt 1 ]; then
  export OMP_PLACES=sockets
fi

srun ./dmem_poisson_amgcl {params}
""")

subprocess.Popen(['sbatch', script_name]).wait()
