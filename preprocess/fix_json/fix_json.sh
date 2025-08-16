#!/bin/bash

# Default output directory
OUTDIR="json_fused_fix"
FILELIST="fault.txt"
# Parse arguments for --out-dir
while [[ $# -gt 0 ]]; do
    case "$1" in
	--output-dir)
	    OUTDIR="$2"
	    shift 2
	    ;;
	--filelist)
	    FILELIST="$2"
	    shift 2
	    ;;
	*)
	    shift
	    ;;
    esac
done
mapfile -t files < $FILELIST
mkdir -p "$OUTDIR"
for ((i=$RANK; i<${#files[@]}; i+=$WORLD_SIZE)); do
  f=${files[$i]}
  [[ -z "$f" ]] && continue
  echo "[rank $RANK/${WORLD_SIZE}] processing: $f"
  python fix_json.py "$f" --out-dir "$OUTDIR" --zstd-threads 32 
  echo "[rank $RANK/${WORLD_SIZE}] Done processing: $f"
done
