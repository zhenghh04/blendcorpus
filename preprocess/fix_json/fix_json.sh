#!/bin/bash
mapfile -t files < fault.txt
# Default output directory
OUTDIR="json_fused_fix"
# Parse arguments for --output_dir
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      OUTDIR="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$OUTDIR"
for ((i=$RANK; i<${#files[@]}; i+=$WORLD_SIZE)); do
  f=${files[$i]}
  [[ -z "$f" ]] && continue
  echo "[rank $RANK/${WORLD_SIZE}] processing: $f"
  python fix_json.py "$f" --out-dir "$OUTDIR" --zstd-threads 32 
  echo "[rank $RANK/${WORLD_SIZE}] Done processing: $f"
done