#!/usr/bin/env python3
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--np',  dest='np', required=True, type=int)
parser.add_argument('--omp', dest='omp', default=1, type=int)
args = parser.parse_args(sys.argv[1:])

script_name = 'scripts/amgcl_ns_{np}x{omp}.sbatch'.format(np=args.np, omp=args.omp)
params = '-A=A.bin -f=b.bin -s=part-{}.mtx -P=spc.json'.format(args.np)

open(script_name, 'w').write(
"""#!/bin/bash
#SBATCH --job-name=amgcl_ns
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
srun ./dmem_ns_amgcl {params}
""".format(np=args.np, omp=args.omp, params=params))

subprocess.Popen(['sbatch', script_name]).wait()
