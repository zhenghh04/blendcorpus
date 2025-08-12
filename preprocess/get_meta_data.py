#!/usr/bin/env python
from blendcorpus.data.gpt_dataset import get_indexed_dataset_
import json
import numpy as np
import glob
import argparse
import os
from tqdm import tqdm
data_impl = 'infer'
skip_warmup = False

def process_list_chunk(my_file_chunk, verbose=0):
    all_data = []
    if verbose == 1:
        my_file_chunk  = tqdm(my_file_chunk)
    for data_prefix in my_file_chunk:
        # Indexed dataset.
        indexed_dataset = get_indexed_dataset_(data_prefix,data_impl,skip_warmup)
        total_num_of_documents = indexed_dataset.sizes.shape[0]
        total_num_of_tokens = np.sum(indexed_dataset.sizes)
        
        data = {
            "data_prefix": data_prefix,
            "total_num_docs": int(total_num_of_documents),
            "total_num_tokens": int(total_num_of_tokens)}
        all_data.append(data)
    return all_data

def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    parser = argparse.ArgumentParser(description="Compute metadata for dataset files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .bin files.")
    parser.add_argument("--output", type=str, default="meta_data.json", help="Path to output JSON file.")
    args = parser.parse_args()

    # Recursively find all .bin files under the input directory
    my_file = [f[:-4] for f in glob.glob(os.path.join(args.input_dir, "**", "*.idx"), recursive=True)]
    if rank == 0:
        print(f"Found {len(my_file)} files; each rank will process {len(my_file)//size}")
    meta_data = process_list_chunk(my_file[rank::size], rank == 0)
    comm.Barrier()    
    if rank == 0:
        print("Gathering all the meta data info from all the ranks...")
    meta_data = comm.gather(meta_data, root=0)
    if rank == 0:
        meta_data = [item for sublist in meta_data for item in sublist]
        json_file_path = args.output
        with open(json_file_path, "w") as json_file:
            json.dump(meta_data, json_file)
if __name__ == "__main__":
    main()
