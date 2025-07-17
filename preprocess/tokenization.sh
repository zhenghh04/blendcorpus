#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--input-dir DIR] [--output-dir DIR] [--num-workers N] [--tokenizer TYPE]"
  echo "  --input-dir    Top-level directory containing .gz files (default: .)"
  echo "  --output-dir   Output directory for tokenized files (default: INPUT_DIR_tok)"
  echo "  --num-workers  Number of workers per file (default: 1)"
  echo "  --tokenizer-type    Tokenizer type to use (default: Llama2Tokenizer)"
  echo "  --tokenizer-model Tokenizer model"
  echo "  --seq-length    sequence length"
  echo "  --append-eod   Append end-of-document token if set (passed to Python script)"
  exit 1
}

## Setting threads in case it blows up
export NUMEXPR_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Default values
INPUT_DIR="."
OUTPUT_DIR=""
NUM_WORKERS=1
SEQ_LENGTH=2048
TOKENIZER_TYPE="Llama2Tokenizer"
FILE_LIST=""
APPEND_EOD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)    INPUT_DIR="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --num-workers)  NUM_WORKERS="$2"; shift 2 ;;
    --tokenizer-type)     TOKENIZER_TYPE="$2"; shift 2 ;;
    --tokenizer-model)    TOKENIZER_MODEL="$2"; shift 2 ;;
    --seq-length)         SEQ_LENGTH="$2"; shift 2 ;;
    --append-eod) APPEND_EOD="--append-eod"; shift ;;
    --file-list)    FILE_LIST="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Set default OUTPUT_DIR if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${INPUT_DIR%/}_tok"
fi

# Create output directory and initialize log file
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/tokenization.log"
{
  echo "Date: $(date)"
  echo "Arguments:"
  echo "  INPUT_DIR=$INPUT_DIR"
  echo "  OUTPUT_DIR=$OUTPUT_DIR"
  echo "  NUM_WORKERS=$NUM_WORKERS"
  echo "  TOKENIZER_TYPE=$TOKENIZER_TYPE"
  echo "  TOKENIZER_MODEL=${TOKENIZER_MODEL}"
  echo "  SEQ_LENGTH=${SEQ_LENGTH}"
  echo "  APPEND_EOD=$APPEND_EOD"
  echo "-------------------------"
} >> "$LOG_FILE"

# MPI-based distribution
if [[ -z "${RANK:-}" || -z "${WORLD_SIZE:-}" ]]; then
  echo "Error: RANK and WORLD_SIZE environment variables must be set by mpiexec." >&2
  exit 2
fi

# Gather all .gz and .zst files
#mapfile -t files < file-list-$PBS_JOBID.txt
if [[ $FILE_LIST == "" ]]; then
    mapfile -t files < <(find -L "$INPUT_DIR" -type f \( -name '*.gz' -o -name '*.zst' -o -name '*.zstd' -o -name '*.json' -o -name '*.jsonl' \))
else
    if [[ $RANK == 0 ]]; then
	echo "Reading files from $FILE_LIST"
    fi
    mapfile -t files < $FILE_LIST
fi


# Filter out files already tokenized
filtered=()
orig_total=${#files[@]}
total=${#files[@]}
if [[ $RANK == 0 ]]; then
    echo "Found $total files"
fi

if [[ $total -lt 100000000 ]]; then
    for ((i=0; i<total; i++)); do
	infile="${files[i]}"
	filename=$(basename "$infile")
	stem=${filename%.gz}
	stem=${stem%.zst}
	stem=${stem%.zstd}	
	stem=${stem%.jsonl}
	stem=${stem%.json}
	relpath="${infile#"$INPUT_DIR"/}"
	outidx="$OUTPUT_DIR/$(dirname "$relpath")/${stem}_text_document.idx"
	if [[ ! -f "$outidx" ]]; then
	    filtered+=("$infile")
	else
	    if [ $RANK -eq 0 ]; then
		echo "$infile is already tokenized"
	    fi
	fi
	# Print progress bar on rank 0 only
	if [[ $RANK -eq 0 ]]; then
	    percent=$(( (i+1)*100 / total ))
	    # Carriage return to overwrite the line
	    printf "\rFiltering files: %d/%d (%d%%)" $((i+1)) $total $percent
	fi
    done
    
    files=("${filtered[@]}")
    

    total=${#files[@]}
    completed=$((orig_total - total))
    if [ $RANK -eq 0 ]; then
	printf "%s\n" "${filtered[@]}" > file-filtered.txt
	echo "Total files: $orig_total, Completed: $completed, Remaining: $total"
    fi
fi
filtered=files
# Process files assigned to this rank
for (( i=$RANK; i<$total; i+=$WORLD_SIZE )); do
  infile="${files[i]}"
  relpath="${infile#"$INPUT_DIR"/}"
  outdir="$OUTPUT_DIR/$(dirname "$relpath")"
  mkdir -p "$outdir"
  filename=$(basename "$infile")
  stem=${filename%.gz}
  stem=${stem%.zst}
  stem=${stem%.zstd}
  stem=${stem%.json}
  stem=${stem%.jsonl}  
  outprefix="$outdir/${stem}"
  if [[ -e ${outprefix}_text_document.idx ]]; then
      echo "${infile} already tokenized"
  else
      RANK=0 WORLD_SIZE=1 preprocess_data --input "$infile" --json-keys text --tokenizer-type "$TOKENIZER_TYPE" --tokenizer-model "$TOKENIZER_MODEL" \
	  --output-prefix "$outprefix" --workers "$NUM_WORKERS" $APPEND_EOD --seq-length $SEQ_LENGTH
  fi
done

#echo "Rank ${RANK}/${WORLD_SIZE} processed $(( (total + WORLD_SIZE - 1 - RANK) / WORLD_SIZE )) files."
