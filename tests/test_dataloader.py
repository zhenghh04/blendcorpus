#!/usr/bin/env python
import os
os.environ['DS_ACCELERATOR']="cpu"
import torch
import time
import json
start_time = time.time()
from mpi4py import MPI
import os
import numpy as np
from blendcorpus import (
    get_config, 
    set_config, 
    mpu, 
    build_gpt_datasets, 
    build_pretraining_data_loader
)


import argparse
import datetime

comm = MPI.COMM_WORLD
def print_rank_0(msg):
    if comm.rank==0:
        print(f" [INFO][{datetime.datetime.now()}] {msg}", flush=True)
end_time = time.time()        
#print_rank_0(f"Loaded python modules in {end_time - start_time} seconds") 

def main():
    ## initialize parallel degree
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel_size=1,
    )

    from blendcorpus.utils import PerfTrace, Profile

    comm.Barrier()
    print_rank_0(f"Barrier synchonization time:  {time.time() - end_time} seconds")
    def get_args():
        parser = argparse.ArgumentParser(description="Test dataloader with MPI")
        parser.add_argument("--trace-dir", dest="trace_dir", required=True,
                            help="Directory to write performance traces")
        parser.add_argument("--data-file-list", dest="data_file_list", required=True,
                            help="Path to the file listing datasets and weights")
        parser.add_argument("--global-batch-size", dest="global_batch_size", type=int, required=True,
                            help="Global batch size for sampling")
        parser.add_argument("--train-iters", dest="train_iters", type=int, required=True,
                            help="Number of training iterations to simulate")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--data-impl", dest="data_impl", default="mmap",
                            help="Data implementation method (e.g., mmap or lazy)") 
        parser.add_argument("--mmap-warmup", action="store_true",
                            help="Whether to perform mmap warmup")
        parser.add_argument("--seq-length", dest="seq_length", type=int, required=True,
                            help="Sequence length for datasets")
        parser.add_argument("--data-cache-path", dest="data_cache_path", default="./data_cache",
                            help="Path to cache data")
        parser.add_argument("--consumed-train-samples", dest="consumed_train_samples", type=int, default=0,
                            help="Number of already consumed train samples")
        parser.add_argument("--consumed-valid-samples", dest="consumed_valid_samples", type=int, default=0,
                            help="Number of already consumed valid samples")
        parser.add_argument("--dataloader-type", dest="dataloader_type",
                            choices=["single", "cyclic"], default="single",
                            help="Type of dataloader to use: 'single' or 'cyclic'")
        parser.add_argument("--micro-batch-size", dest="micro_batch_size",
                            type=int, default=4,
                            help="Micro batch size for each data parallel rank")
        parser.add_argument("--data-sharding", dest="data_sharding",
                            action="store_true",
                            help="Enable data sharding for cyclic sampler")
        parser.add_argument("--num-workers", dest="num_workers",
                            type=int, default=2,
                            help="Number of worker processes for DataLoader")
        parser.add_argument("--multiprocessing-context", dest="multiprocessing_context",
                            default="fork",
                            help="Multiprocessing context for DataLoader workers")
        parser.add_argument("--repeated-dataloader", dest="repeated_dataloader",
                            action="store_true",
                            help="Wrap DataLoader in RepeatingLoader if set")
        parser.add_argument('--blend-sample-in-corpus', action="store_true", help="whether to blend samples in corpus")
        parser.add_argument('--shuffle-sample-in-corpus', action="store_true", help="whether to shuffle samples in corpus")
        parser.add_argument('--print-sample-info', action='store_true')
        parser.add_argument('--dataloader-iter', action='store_true')
        return parser.parse_args()

    args = get_args()
    set_config(args)
    config = get_config()
    if comm.rank == 0:
        print(config)
    os.makedirs(args.trace_dir, exist_ok=True)
    # Build datasets
    start_build_dataset = time.time()
    print_rank_0(f"Starting to build the blendable dataset")
    train_ds, valid_ds, test_ds = build_gpt_datasets(config)
    end_build_dataset = time.time()
    print_rank_0(f"Finished building the blendable dataset in {end_build_dataset - start_build_dataset} second")
    print_rank_0(f"Total number of samples: {len(train_ds)} {len(valid_ds) if valid_ds is not None else 0} {len(test_ds) if test_ds is not None else 0}")

    def get_sample_info(blendable_dataset, idx):
        # corpus dataset
        cd = blendable_dataset.dataset_index[idx]
        # index within the corpus dataset
        cds = blendable_dataset.dataset_sample_index[idx]
        # dataset index within each corpus
        shuffle_idx = blendable_dataset.datasets[cd].shuffle_index[cds]
        fcd = blendable_dataset.datasets[cd].dataset_index[shuffle_idx]
        # sample index within the dataset
        fcds = blendable_dataset.datasets[cd].dataset_sample_index[shuffle_idx]
        # corresponding data file
        prefix = blendable_dataset.datasets[cd].dataset_builders[fcd].prefix
        corpus = blendable_dataset.datasets[cd].dataset_builders[fcd].corpus
        return prefix, corpus, fcds

    #-------
    files = []
    weights = []
    flist = []
    corpus_all = []
    with open(args.data_file_list, 'r') as fin:
        for f in fin.readlines():
            w, fname, c = f.split()
            weights.append(float(w))
            flist.append(fname)
            files.append(float(w))
            files.append(fname)
            files.append(c)
            if c not in corpus_all:
                corpus_all.append(c)
    # ---- 
    if args.print_sample_info:
        if comm.rank == 0:
            fout = open("samples_list.jsonl", "w")        
            for i in range(args.train_iters):
                ns_corpus = {}
                for c in corpus_all:
                    ns_corpus[c] = 0
                for j in range(args.global_batch_size):
                    prefix, corpus, idx = get_sample_info(train_ds, i*args.global_batch_size+j)
                    ns_corpus[corpus] +=1
                    fout.write(f"\u007b 'batch': {i}, 'sample': {j}, 'corpus': '{corpus}', 'prefix': '{prefix}', 'dataset_sample_index': {idx} \u007d\n")
                fout.write(f"\u007b 'batch': {i}, 'histogram': {ns_corpus} \u007d \n")

    comm.Barrier()        
    start_build_dataloader = time.time()
    print_rank_0(f"Starting to build the data loader")
    rank_in_parallel_group = mpu.get_sequence_parallel_rank()

    print_rank_0(f"rank_in_parallel_group: {rank_in_parallel_group}")
    train_dataloader = build_pretraining_data_loader(
        train_ds, args.consumed_train_samples, config)
    print_rank_0(f"train done")
    valid_dataloader = build_pretraining_data_loader(
        valid_ds, args.consumed_valid_samples, config)
    test_dataloader = build_pretraining_data_loader(test_ds, 0, config)
    end_build_dataloader = time.time()
    print_rank_0(f"Finished building the data loader in {end_build_dataloader - start_build_dataloader} second")

    print_rank_0(f"Starting loading the data")
    start_loading_time = time.time()
    NUM_ITEMS=1
    SLEEP_TIME=10.0
    def compute(ct):
        time.sleep(ct)
    n=0
    if args.dataloader_iter:
        start_time = time.time()
        for i in iter(train_dataloader):
            print(f"[{comm.rank}] DATA {i}")
            n+=1
            if (n%NUM_ITEMS==0):
                print_rank_0(f"Proccessed {n}th-batch in {time.time() - start_time}")
            if n>=1000:
                break
            start_time = time.time()
        end_loading_time = time.time()
        print_rank_0(f"Finished loading the data ({n} batches) in {end_loading_time - start_loading_time}")


# Entrypoint for script execution
if __name__ == "__main__":
    from blendcorpus.dist_setup import init_distributed
    init_distributed()
    main()
