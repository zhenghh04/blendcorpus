#!/usr/bin/env bash
#PBS -A datascience
#PBS -l walltime=5:00:00
#PBS -l select=1
#PBS -l filesystems=home:eagle
cd $PBS_O_WORKDIR
source $HOME/crux/AuroraGPT/conda.sh
set -euo pipefail
# Invoke the test with mpirun
export PPN=16
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
touch completed_0.txt
cat completed_*.txt | sort -u > completed.txt
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN --cpu-bind depth -d 4 launcher.sh ./fuse_files_parallel.sh
