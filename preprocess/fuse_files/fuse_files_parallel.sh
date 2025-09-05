#!/usr/bin/env bash
set -euo pipefail

# ---- params (env-overridable) ----
ROOT="${ROOT:-data}"                 # input root
OUT="${OUT:-data_fused}"             # output root
MAX=$((5*1024*1024*1024))            # target ~5 GiB compressed per fused file
THREADS_PER_RANK="${THREADS_PER_RANK:-4}"    # zstd threads per rank
ZSTD_LEVEL="${ZSTD_LEVEL:--9}"       # compression level
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

mkdir -p "$OUT"

print_rank_0() {
    if [[ $RANK == 0 ]]; then
	echo $1
    fi
}

export OMP_NUM_THREADS=1
nice -n 5 ionice -c2 -n5 true 2>/dev/null || true

# ------------------ helpers ------------------
decompress_stream() {
  # Decompress each input (paths on stdin) to stdout.
  # Insert a single "\n" *only if* the previous file's last byte wasn't a newline.
  local first=1
  local last_char=""
  while IFS= read -r f; do
    case "$f" in
	*.jsonl.zstd|*.json.zstd|*.json.zst|*.jsonl.szt)
	    cmd=(zstdcat "$f") ;;
	*.json.gz|*.jsonl.gz)
            if command -v pigz >/dev/null 2>&1; then cmd=(pigz -dc "$f"); else cmd=(gzip -dc "$f"); fi
            ;;
	*.json|*.jsonl) cmd=(cat "$f") ;;
	*) echo "Unknown format: $f" >&2; exit 2 ;;
    esac

    # Only add a separator if not first and prior file didn't end with a newline
    if (( ! first )) && [[ "$last_char" != $'\n' ]]; then
      printf '\n'
    fi

    # Stream once, but capture the last byte via process substitution so we
    # don't have to decompress twice.
    local _tmp; _tmp="$(mktemp)"
    # shellcheck disable=SC2069
    "${cmd[@]}" \
      | tee >(tail -c 1 > "$_tmp")
    last_char="$(cat "$_tmp" 2>/dev/null || true)"
    rm -f "$_tmp"

    first=0
  done
}

make_groups_file() {
  # $1 = tsv (path<TAB>size)
  # stdout: one line per group with space-separated file paths (no empty groups)
  awk -F'\t' -v MAX="$MAX" '
    {
      f=$1; sz=$2+0;
      if (cur>0 && cur+sz>MAX) { print group; group=""; cur=0 }
      group = (group=="" ? f : group " " f);
      cur += sz;
    }
    END { if (group!="") print group }
  ' "$1"
}

process_all_subfolders_distributed() {
    # Build groups PER SUBFOLDER, then distribute all groups (from all subfolders)
    # evenly across ranks. Each fused output stays within its original subfolder.
    local global_out="$OUT/_global"
    mkdir -p $OUT/dummy/
    touch $OUT/dummy/completed_0.txt
    COMPLETE_ALL=$OUT/_global/completed.txt
    mkdir -p "$global_out"    
    barrier.sh
    if [[ $RANK == 0 ]]; then
	rm -f $OUT/_global/completed.txt
	touch $OUT/_global/completed.txt
	cat $OUT/*/completed_*.txt >& $COMPLETE_ALL
	cat $COMPLETE_ALL
    fi
    barrier.sh
  local master_groups="$global_out/groups_all.tsv"
  : > "$master_groups"

  # Enumerate immediate subfolders under $ROOT
  mapfile -t SUBS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort -V)
  print_rank_0 "[rank $RANK/$WORLD_SIZE] Found ${#SUBS[@]} subfolders under '$ROOT'"


  # Build per-subfolder groups and append to master list as: subfolder<TAB>local_idx<TAB>local_count<TAB>files...
  for sub in "${SUBS[@]}"; do
    local subdir="$ROOT/$sub"
    local outdir="$OUT/$sub"
    mkdir -p "$outdir"

    local tsv="$outdir/files_and_sizes.tsv"
    find "$subdir" -type f \( -name '*.jsonl.zstd' -o -name '*.json.gz' -o -name "*.parquet" -o -name '*.json*' \) \
        -printf '%p\t%s\n' | sort -V > "$tsv"
    if [[ ! -s "$tsv" ]]; then
	print_rank_0 "  [rank $RANK] $sub: no matching files"; continue
    fi

    local groups="$outdir/groups.txt"
    if [[ $RANK == 0 ]]; then
	# only rank zero write the files
	make_groups_file "$tsv" > "$groups"
	sed -i '/^$/d' "$groups" || true
	if [[ ! -s "$groups" ]]; then
	    echo "  [rank $RANK] $sub: no non-empty groups"; continue
	fi

    
    local NG_SUB; NG_SUB=$(wc -l < "$groups" | tr -d '[:space:]')
    local idx=0
    while IFS= read -r line; do
	[[ -z "$line" ]] && continue
	idx=$((idx+1))
	printf '%s\t%d\t%d\t%s\n' "$sub" "$idx" "$NG_SUB" "$line" >> "$master_groups"
    done < "$groups"
    fi    
  done
  barrier.sh
  if [[ ! -s "$master_groups" ]]; then
    echo "[rank $RANK] global: no groups built; nothing to do."; return
  fi

  local NGROUPS; NGROUPS=$(wc -l < "$master_groups" | tr -d '[:space:]')
  print_rank_0 "[rank $RANK] global: total groups across subfolders = $NGROUPS"

  # Contiguous block assignment across all groups
  #local per_rank=$(( (NGROUPS + WORLD_SIZE - 1) / WORLD_SIZE ))
  #local start=$(( RANK*per_rank + 1 ))
  #local end=$(( start + per_rank - 1 ))
  #if (( end > NGROUPS )); then end=$NGROUPS; fi
  #if (( start > NGROUPS )); then
  #  echo "  [rank $RANK] global: no assigned groups (start>$NGROUPS)"; return
  #fi
  #echo "[rank $RANK] global: assigned groups $start..$end"

  # Process only this rank's slice
  local i=0
  while IFS=$'\t' read -r sub idx ng_sub files; do
      i=$((i+1))      
      if (( (i-1) % WORLD_SIZE != RANK )); then
	  continue
      else
	  echo "[RANK-$RANK] processing $i"
      fi
    local outdir="$OUT/$sub"
    mkdir -p "$outdir"
    local completed="$outdir/completed_$RANK.txt"
    touch "$completed"
    # Get extension from first file in group
    local firstfile
    local tmplist; tmplist="$(mktemp)"
    tr ' ' '\n' <<< "$files" | sed '/^$/d' > "$tmplist"
    
    firstfile=$(awk 'NR==1{print; exit}' "$tmplist")
    local ext
    echo $firstfile
    case "$firstfile" in
	*.jsonl.zst|*.jsonl.zstd) ext="jsonl.zstd" ;;
	*.json.zst|*.json.zstd) ext="json.zstd" ;;
	*.json) ext="json";;
	*.jsonl) ext="jsonl";;	
	*.json.gz) ext="json.gz" ;;
	*.jsonl.gz) ext="jsonl.gz" ;;	
	*.parquet) ext="parquet" ;;
	*) echo "  [rank $RANK] ERROR: unknown file extension for $firstfile"; exit 1 ;;
    esac

    # Set outfile accordingly
    local outfile
    outfile=$(printf "%s/fused_%04d_of_%04d.%s" "$outdir" "$idx" "$ng_sub" "$ext")

    #local outfile; outfile=$(printf "%s/fused_%04d_of_%04d.jsonl.zstd" "$outdir" "$idx" "$ng_sub")
    if grep -Fxq "$outfile" "$COMPLETE_ALL"; then
      echo "  [rank $RANK] $sub: exists, skip $(basename "$outfile")"
      continue
    fi

    echo "  [rank $RANK] $sub: fuse group $idx/$ng_sub -> $(basename "$outfile")"

    # Write file list for this group to a temp file and fuse it
    local tmplist; tmplist="$(mktemp)"
    tr ' ' '\n' <<< "$files" | sed '/^$/d' > "$tmplist"
    if [[ ! -s "$tmplist" ]]; then
      echo "  (empty group, skipping)"; rm -f "$tmplist"; continue
    fi
    case "$outfile" in
	*.json*)
	    decompress_stream < "$tmplist" \
		| zstd -q -f -T"$THREADS_PER_RANK" "$ZSTD_LEVEL" -o "$outfile"
	    ;;
	*.parquet)
	    merge_parquet "$tmplist" --output "$outfile"
	    ;;
	*) echo "Unknown format: $outfile" >&2; exit 2 ;;	
    esac    
    rm -f "$tmplist"
    echo "$outfile" >> "$completed"
  done < "$master_groups"
  barrier.sh
}

process_all_global() {
  # Build one global list of candidate files from all subfolders under $ROOT,
  # group them to ~MAX bytes per fused shard, then distribute *contiguous* groups
  # evenly across ranks. All outputs are written under "$OUT/global".
  local outdir="$OUT/global"
  mkdir -p "$outdir"
  local completed="$outdir/completed_$RANK.txt"
  touch "$completed"

  echo "[rank $RANK] global: building file list under '$ROOT'"
  local tsv="$outdir/files_and_sizes_global.tsv"
  find "$ROOT" -type f \( -name '*.jsonl.zstd' -o -name '*.json.gz' \) \
      -printf '%p\t%s\n' | sort -V > "$tsv"
  if [[ ! -s "$tsv" ]]; then
    echo "  [rank $RANK] global: no matching files"
    return
  fi

  local groups="$outdir/groups_global.txt"
  make_groups_file "$tsv" > "$groups"
  sed -i '/^$/d' "$groups" || true
  if [[ ! -s "$groups" ]]; then
    echo "  [rank $RANK] global: no non-empty groups"
    return
  fi

  local NGROUPS; NGROUPS=$(wc -l < "$groups" | tr -d '[:space:]')
  echo "[rank $RANK] global: total groups = $NGROUPS"

  # Equally distribute groups to ranks as contiguous blocks
  local per_rank=$(( (NGROUPS + WORLD_SIZE - 1) / WORLD_SIZE ))  # ceil
  local start=$(( RANK*per_rank + 1 ))
  local end=$(( start + per_rank - 1 ))
  if (( end > NGROUPS )); then end=$NGROUPS; fi
  if (( start > NGROUPS )); then
    echo "  [rank $RANK] global: no assigned groups (start>$NGROUPS)"; return
  fi
  echo "[rank $RANK] global: assigned groups $start..$end"

  # Fuse assigned groups
  local i=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && { i=$((i+1)); continue; }
    i=$((i+1))
    if (( i < start || i > end )); then continue; fi
    local outfile; outfile=$(printf "%s/fused_%04d_of_%04d.jsonl.zstd" "$outdir" "$i" "$NGROUPS")
    if grep -Fxq "$outfile" "$completed"; then
      echo "  [rank $RANK] global: exists, skip $(basename "$outfile")"
      continue
    fi
    echo "  [rank $RANK] global: fuse $i/$NGROUPS -> $(basename "$outfile")"
    fuse_group_from_line "$line" "$outfile"
    echo "$outfile" >> "$completed"
  done < "$groups"
}

fuse_group_from_line() {
  # $1 = "file1 file2 ..."
  # $2 = outfile
  local line="$1" out="$2"
  local tmplist
  tmplist="$(mktemp)"
  tr ' ' '\n' <<< "$line" | sed '/^$/d' > "$tmplist"
  if [[ ! -s "$tmplist" ]]; then
    echo "  (empty group, skipping)"; rm -f "$tmplist"; return 0
  fi
  decompress_stream < "$tmplist" \
      | zstd -q -f -T"$THREADS_PER_RANK" "$ZSTD_LEVEL" -o "$out"
  rm -f "$tmplist"
}

process_subfolder_single() {
  local subname="$1"
  local subdir="$ROOT/$subname"
  local outdir="$OUT/$subname"
  mkdir -p "$outdir"

  local tsv="$outdir/files_and_sizes.tsv"
  find "$subdir" -type f \( -name '*.jsonl.zstd' -o -name '*.json.gz' \) \
      -printf '%p\t%s\n' | sort -V > "$tsv"
  if [[ ! -s "$tsv" ]]; then
    echo "  [rank $RANK] $subname: no matching files"
    return
  fi

  local groups="$outdir/groups.txt"
  make_groups_file "$tsv" > "$groups"
  sed -i '/^$/d' "$groups" || true
  if [[ ! -s "$groups" ]]; then
    echo "  [rank $RANK] $subname: no non-empty groups"
    return
  fi
  local NGROUPS; NGROUPS=$(wc -l < "$groups" | tr -d '[:space:]')

  local i=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    i=$((i+1))
    local outfile; outfile=$(printf "%s/fused_%04d_of_%04d.jsonl.zstd" "$outdir" "$i" "$NGROUPS")
    if grep -Fxq "$outfile" completed.txt; then
      echo "  [rank $RANK] $subname: exists, skip $(basename "$outfile")"
      continue
    fi
    echo "  [rank $RANK] $subname: fuse $i/$NGROUPS -> $(basename "$outfile")"
    fuse_group_from_line "$line" "$outfile"
    echo "$outfile" >> completed_$RANK.txt
  done < "$groups"
}

process_folder_distributed() {
  local subdir="$ROOT/"
  local outdir="$OUT/"
  mkdir -p "$outdir"
  local completed="$outdir/completed_$RANK.txt"
  touch "$completed"

  echo "[rank $RANK] dclm: building global list"
  local tsv="$outdir/files_and_sizes_$RANK.tsv"
  find "$subdir" -type f \( -name '*.jsonl.zstd' -o -name '*.json.gz' \) \
      -printf '%p\t%s\n' | sort -V > "$tsv"
  if [[ ! -s "$tsv" ]]; then
    echo "  [rank $RANK] dclm: no matching files"
    return
  fi

  local groups="$outdir/groups_global_$RANK.txt"
  make_groups_file "$tsv" > "$groups"
  sed -i '/^$/d' "$groups" || true
  if [[ ! -s "$groups" ]]; then
    echo "  [rank $RANK] dclm: no non-empty groups"
    return
  fi
  local NGROUPS; NGROUPS=$(wc -l < "$groups" | tr -d '[:space:]')
  echo "[rank $RANK] total groups = $NGROUPS"

  # Equally distribute groups to ranks as contiguous blocks
  local per_rank=$(( (NGROUPS + WORLD_SIZE - 1) / WORLD_SIZE ))  # ceil
  local start=$(( RANK*per_rank + 1 ))
  local end=$(( start + per_rank - 1 ))
  if (( end > NGROUPS )); then end=$NGROUPS; fi
  if (( start > NGROUPS )); then
    echo "  [rank $RANK] no assigned groups (start>$NGROUPS)"; return
  fi
  echo "[rank $RANK] assigned groups $start..$end"

  # Fuse assigned groups
  local i=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && { i=$((i+1)); continue; }
    i=$((i+1))
    if (( i < start || i > end )); then continue; fi
    local outfile; outfile=$(printf "%s/fused_%04d_of_%04d.jsonl.zstd" "$outdir" "$i" "$NGROUPS")
    if grep -Fxq "$outfile" "$completed"; then
      echo "  [rank $RANK] exists, skip $(basename "$outfile")"
      continue
    fi
    echo "  [rank $RANK] fuse $i/$NGROUPS -> $(basename "$outfile")"
    fuse_group_from_line "$line" "$outfile"
    echo "$outfile" >> "$completed"
  done < "$groups"
}

# ------------------ main flow ------------------
# Build groups per subfolder and distribute globally across ranks

num_subfolders=$(find $ROOT -mindepth 1 -maxdepth 1 -type d | wc -l)
if [[ $num_subfolders == 0 ]]; then
  process_folder_distributed
else
  process_all_subfolders_distributed
fi
barrier.sh
echo "[rank $RANK] done."
