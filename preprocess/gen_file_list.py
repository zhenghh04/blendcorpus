#!/usr/bin/env python3
# Argonne National Laboratory, 2025. All rights reserved.
# Usage: python gen_file_list.py --input-json data_fused_gemma_eod.json --output olmo-fused-file-list.txt --topdir /flare/AuroraGPT/datasets/olmo-mix-1124/ --epochs 1 1 1

import argparse
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate category token weights from JSON data list.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="data_fused_gemma_eod.json",
        help="Path to the input JSON file containing the data list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="olmo-fused-file-list.txt",
        help="Path to the output file where results will be written",
    )
    parser.add_argument(
        "--topdir",
        type=str,
        default="/flare/AuroraGPT/datasets/olmo-mix-1124/",
        help="Top directory prefix to use in output file list",
    )
    parser.add_argument(
        '--epochs',
        nargs='+',
        type=float,
        help='A list of float for epochs'
    )
    args = parser.parse_args()

    # Load your data list
    with open(args.input_json, "r") as fin:
        data = json.load(fin)
    print(f"Loaded {len(data)} items...")
    df = pd.DataFrame(data)

    # Extract category between the first and second '/'
    df['category'] = df['data_prefix'].apply(lambda x: x.split('/')[1])
    corpora=list(df['category'].unique())    
    print("Corpora:", corpora)

    # Sum tokens
    grouped = df.groupby('category', as_index=False)['total_num_tokens'].sum()
    nc = len(grouped['category'])
    total = 0
    epochs = {}    
    if args.epochs is None:
        print(f"please specify --epochs for all the corpora to generate the file list in following order {corpora}")
        for i in range(len(corpora)):
            epochs[corpora[i]] = 1
    else:
        print(f"Epochs set: {args.epochs}")
        for i in range(len(corpora)):
            epochs[corpora[i]] = args.epochs[i]
    print("\n")
    print("%20s %15s %5s %15s" %("Corpus", "#tokens", "Epoch", "#tokens(taken)"))
    print("-"*80)
    for i in range(nc):
        c = grouped['category'][i]
        ep = epochs[c]
        total += int(grouped['total_num_tokens'][i] * ep)
        print(f"{grouped['category'][i]:20s}  {grouped['total_num_tokens'][i]:15d}  {ep:4.1f}  {int(grouped['total_num_tokens'][i]*ep):15d}")
    print("-"*80)
    print(f"Total number of tokens: {total: 15d}")

    nf = len(df['category'])
    print(f"Total number of files: {nf}")

    # Write file list
    with open(args.output, "w") as fout:
        i = 0
        for prefix in df['data_prefix']:
            fout.write("%10.10f %s %s\n"%(float(df['total_num_tokens'][i]*ep)/total, f"{args.topdir}{df['data_prefix'][i]}", df['category'][i]))
            i+=1

if __name__ == "__main__":
    main()
