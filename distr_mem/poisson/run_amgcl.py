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

script_name = 'scripts/amgcl_{np}x{omp}_{dpp}'.format(
        np=args.np, omp=args.omp, dpp=args.dpp)
params = 'isolver.maxiter=500 local.coarse_enough=500 local.coarsening.aggr.eps_strong=0.0 -i=cg -d=eigen_splu'

if args.strong:
    params += ' -n={}'.format(args.dpp)
else:
    params += ' -n={}'.format(int(0.5 + (args.np * args.omp * args.dpp)**(1/3)))

if args.const:
    script_name += '_const'
    params += ' --cd'

script_name += '.sbatch'

open(script_name, 'w').write(
"""#!/bin/bash
#SBATCH --job-name=amgcl
#SBATCH -D .
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task={omp}
#SBATCH --ntasks={np}
#SBATCH --time=00:10:00

export OMP_NUM_THREADS={omp}
export OMP_WAIT_POLICY=ACTIVE

if [ {omp} -gt 1 ]; then
  export OMP_PLACES=sockets
fi

echo "params: {params}"
srun ./dmem_poisson_amgcl {params}
""".format(np=args.np, omp=args.omp, params=params))

subprocess.Popen(['sbatch', script_name]).wait()
