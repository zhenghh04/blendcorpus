# Fusing Small Files for Efficient Processing

This directory contains scripts and tools to fuse many small files into larger ones. Fusing small files is important because it reduces the index-building overhead in Megatron-DeepSpeed and helps avoid rounding errors that can occur when processing very small files. By combining these files, we can improve efficiency and stability during training or data preprocessing.

## How to Run `fuse_files_parallel.sh`

The `fuse_file_parallel.sh` script allows you to fuse files in parallel, leveraging multiple processes to speed up the operation. It supports environment variables such as `WORLD_SIZE` (total number of parallel workers), `RANK` (the rank of the current worker), and others that help coordinate the parallel execution.

### Usage Example

```bash
export PPN=16
export THREADS_PER_RANK=4
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN --cpu-bind depth -d $THREADS_PER_RANK launcher.sh \
  ./fuse_files_parallel.sh --input-dir /path/to/data --output-dir /path/to/data-fused
```
Example submission scripts can be found in [./fuse.sh](./fuse.sh)

Make sure to adjust the environment variables accordingly for each parallel worker.

### Dry Run (Count Only)

Use `--dry-run` to predict how many fused files will be created, without writing outputs.

Single-process:
```bash
./fuse_files_parallel.sh --input-dir /path/to/data --dry-run
```

MPI:
```bash
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN --cpu-bind depth -d $THREADS_PER_RANK launcher.sh ./fuse_files_parallel.sh --dry-run
```
In MPI mode, rank 0 performs counting and other ranks exit.

## Recent Improvement

The fusion script has been improved to ensure that newlines are only inserted between files if needed. This prevents JSON objects from being glued together without introducing unnecessary empty lines, maintaining proper JSON formatting and avoiding parsing errors.
