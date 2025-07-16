#!/usr/bin/env python3
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Mix multiple file-lists, normalize internal weights, and apply global file weights."
    )
    p.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help="Pairs of file_list and global_weight, e.g.: --inputs 0.3  f1.txt 0.7 f2.tx"
    )
    return p.parse_args()


def main():
    args = parse_args()
    inp = args.inputs
    if len(inp) % 2 != 0:
        sys.exit("Error: --inputs must be an even number of arguments (file weight pairs).")

    # Group into (file_path, global_weight)
    pairs = []
    for i in range(0, len(inp), 2):
        file_path = inp[i+1]
        try:
            gw = float(inp[i])
        except ValueError:
            sys.exit(f"Error: global weight must be a number, got '{inp[i]}'")
        if gw <= 0:
            sys.exit(f"Error: global weight must be positive, got {gw}")
        pairs.append((gw, file_path))

    # Compute sum of all global weights (if normalization across files is desired)
    sum_global = sum(gw for gw, _ in pairs)

    for gw, file_path in pairs:
        # Normalized file-level fraction (optional across all files)
        file_fraction = gw / sum_global

        # Read entries and sum file-local weights
        entries = []
        file_sum = 0.0
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        sys.exit(f"Error: each line must have prefix weight corpus, got: '{line}'")
                    prefix = parts[1]
                    try:
                        w = float(parts[0])
                    except ValueError:
                        sys.exit(f"Error: weight must be numeric, got '{parts[1]}' in file {file_path}")
                    corpus = parts[2]
                    entries.append((prefix, w, corpus))
                    file_sum += w
        except FileNotFoundError:
            sys.exit(f"Error: cannot open file '{file_path}'")

        if file_sum <= 0:
            sys.exit(f"Error: sum of weights in file '{file_path}' is non-positive: {file_sum}")

        # Print header only once
        # Compute and print normalized weights
        for prefix, w, corpus in entries:
            new_w = (w / file_sum) * file_fraction
            print(f"{new_w:.6f} {prefix} {corpus}")

if __name__ == '__main__':
    main()
