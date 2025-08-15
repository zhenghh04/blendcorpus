#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -A datascience
#PBS -l filesystems=eagle:home

cd $PBS_O_WORKDIR
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
source ../conda.sh
mpiexec -np $PBS_JOBSIZE --ppn 1 --cpu-bind depth -d 64 launcher.sh bash ./fix_json.sh --output_dir fused_json_fix
