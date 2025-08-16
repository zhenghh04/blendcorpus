#!/bin/bash
# Try PMIX environment first
if [[ -n "$PMIX_LOCAL_RANK" ]]; then
    export LOCAL_RANK=$PMIX_LOCAL_RANK
    export RANK=$PMIX_RANK
elif [[ -n "$PALS_LOCAL_RANKID" ]]; then
    # Fall back to PALS
    export LOCAL_RANK=$PALS_LOCAL_RANKID
    export RANK=$PALS_RANKID
fi

# Determine WORLD_SIZE
if [[ -n "$PBS_NODEFILE" && -f "$PBS_NODEFILE" ]]; then
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
else
    export PBS_JOBSIZE=1
fi

if [[ -n "$PALS_LOCAL_SIZE" ]]; then
    export WORLD_SIZE=$((PALS_LOCAL_SIZE * PBS_JOBSIZE))
elif [[ -n "$SLURM_NTASKS" ]]; then
    export WORLD_SIZE=$SLURM_NTASKS
elif [[ -n "$PMI_SIZE" ]]; then
    export WORLD_SIZE=$PMI_SIZE
else
    export WORLD_SIZE=1
fi

# Set default values if not defined
export RANK=${RANK:-${SLURM_PROCID:-${PMI_RANK:-0}}}
export LOCAL_RANK=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
echo "I am $RANK of $WORLD_SIZE"
$@