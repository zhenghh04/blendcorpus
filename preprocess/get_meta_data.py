#!/usr/bin/env python
from blendcorpus.data.gpt_dataset import get_indexed_dataset_
from blendcorpus import print_rank_0
import json
import numpy as np
import glob
from mpi4py import MPI
import math
import argparse
import os
import tqdm 

def main():
    data_impl = 'infer'
    skip_warmup = False
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Gather token metadata from .bin files using MPI")
    parser.add_argument("--input-dir", required=True,
                        help="Top directory containing tokenized .bin files (searches recursively)")
    parser.add_argument("--output", required=True,
                        help="Path to the output JSON file")
    args = parser.parse_args()
    def process_list_chunk(my_file_chunk):
        all_data = []
        if rank == 0:
            my_file_chunk = tqdm.tqdm(my_file_chunk)
        for data_prefix in my_file_chunk:
            # Indexed dataset.
            print_rank_0(data_prefix)
            if not os.path.isfile(f"{data_prefix}.idx"):
                print_rank_0(f"Warning: {data_prefix} tokenization is not finished")
                continue
            indexed_dataset = get_indexed_dataset_(data_prefix,data_impl,skip_warmup)
            print(indexed_dataset)
            total_num_of_documents = indexed_dataset.sizes.shape[0]
            total_num_of_tokens = np.sum(indexed_dataset.sizes)
            
            data = {
                "data_prefix": data_prefix,
                "total_num_docs": int(total_num_of_documents),
                "total_num_tokens": int(total_num_of_tokens)}
            print_rank_0(data)
            all_data.append(data)
        return all_data
    # Recursively find all .bin files under the input directory
    pattern = os.path.join(args.input_dir, "**", "*.bin")
    my_file = [os.path.splitext(f)[0] for f in glob.glob(pattern, recursive=True)]
    # Partition the file list among MPI ranks
    total_files = len(my_file)
    my_chunk = my_file[rank::size]

    # Each rank processes its chunk
    local_meta = process_list_chunk(my_chunk)

    # Gather results at root
    all_meta = comm.gather(local_meta, root=0)

    # Only root writes the combined JSON
    if rank == 0:
        # Flatten the list of lists
        flat_meta = [item for sublist in all_meta for item in sublist]
        flat_meta.sort(key=lambda x: x["data_prefix"])
        json_file_path = args.output
        with open(json_file_path, "w+") as json_file:
            json.dump(flat_meta, json_file)
        print(f"Wrote metadata for {len(flat_meta)} files to {json_file_path}")
    # Ensure MPI finalizes cleanly
    comm.Barrier()

if __name__ == "__main__":
    main()
