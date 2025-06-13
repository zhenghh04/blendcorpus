#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--input-dir DIR] [--output-dir DIR] [--num-workers N] [--tokenizer TYPE]"
  echo "  --input-dir    Top-level directory containing .gz files (default: .)"
  echo "  --output-dir   Output directory for tokenized files (default: INPUT_DIR_tok)"
  echo "  --num-workers  Number of workers per file (default: 1)"
  echo "  --tokenizer-type    Tokenizer type to use (default: Llama2Tokenizer)"
  echo "  --tokenizer-model Tokenizer model"
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
TOKENIZER_TYPE="Llama2Tokenizer"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)    INPUT_DIR="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --num-workers)  NUM_WORKERS="$2"; shift 2 ;;
    --tokenizer-type)     TOKENIZER_TYPE="$2"; shift 2 ;;
    --tokenizer-model)    TOKENIZER_MODEL="$2"; shift 2 ;; 
    -h|--help)      usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Set default OUTPUT_DIR if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${INPUT_DIR%/}_tok"
fi

# MPI-based distribution
if [[ -z "${RANK:-}" || -z "${WORLD_SIZE:-}" ]]; then
  echo "Error: RANK and WORLD_SIZE environment variables must be set by mpiexec." >&2
  exit 2
fi

# Gather all .gz and .zst files
mapfile -t files < <(find "$INPUT_DIR" -type f \( -name '*.gz' -o -name '*.zst' \))
# Filter out files already tokenized
filtered=()
orig_total=${#files[@]}

for infile in "${files[@]}"; do
  filename=$(basename "$infile")
  stem=${filename%.gz}
  stem=${stem%.zst}
  stem=${stem%.jsonl}
  stem=${stem%.json}
  relpath="${infile#"$INPUT_DIR"/}"
  outidx="$OUTPUT_DIR/$(dirname "$relpath")/${stem}_text_document.idx"
  if [[ ! -f "$outidx" ]]; then
    filtered+=("$infile")
  fi
done
files=("${filtered[@]}")
total=${#files[@]}
completed=$((orig_total - total))
if [ $RANK -eq 0 ]; then
    echo "Total files: $orig_total, Completed: $completed, Remaining: $total"
fi
# Process files assigned to this rank
for (( i=$RANK; i<$total; i+=$WORLD_SIZE )); do
  infile="${files[i]}"
  relpath="${infile#"$INPUT_DIR"/}"
  outdir="$OUTPUT_DIR/$(dirname "$relpath")"
  mkdir -p "$outdir"
  filename=$(basename "$infile")
  stem=${filename%.gz}
  stem=${stem%.zst}
  stem=${stem%.json}
  stem=${stem%.jsonl}  
  outprefix="$outdir/${stem}"
  preprocess_data --input "$infile" --json-keys text --tokenizer-type "$TOKENIZER_TYPE" --tokenizer-model "$TOKENIZER_MODEL" \
    --output-prefix "$outprefix" --workers "$NUM_WORKERS"
done

echo "Rank ${RANK}/${WORLD_SIZE} processed $(( (total + WORLD_SIZE - 1 - RANK) / WORLD_SIZE )) files."
