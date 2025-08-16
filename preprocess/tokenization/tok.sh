#!/bin/bash
#PBS -A datascience
#PBS -l filesystems=home:eagle
#PBS -l select=8
#PBS -l walltime=4:00:00
#PBS -q workq
#PBS -N tok-eod

export PALS_RPC_TIMEOUT=3600
cd $PBS_O_WORKDIR

# important to use PPN = 1, and set larger number of threads
export PPN=1
export NUM_WORKERS=64

export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export PREPROCESS=tokenization.sh

# Change the tokenizer file
#  - for HFTokenizer, one has to provide the tokenizer directory
#  - for others, one provide the path to tokenizer.model

export TOKENIZER=/home/hzheng/AuroraGPT/olmo-mix-1124/gemma-7b/

# input folder
export DATA=data

# output folder
export DATA_TOK=data_tok

# tokenization only need CPU. 
export DS_ACCELERATOR=cpu

mpiexec -n $((PBS_JOBSIZE*PPN)) --ppn $PPN  --cpu-bind depth -d $NUM_WORKERS \
       launcher.sh $PREPROCESS \
       --input-dir $DATA \
       --output-dir $DATA_TOK \
       --num-workers $NUM_WORKERS \
       --tokenizer-type HFTokenizer \
       --append-eod \
       --tokenizer-model $TOKENIZER \
       --seq-length 10000000000000
