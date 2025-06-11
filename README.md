
```bash
mpiexec -n $((PBS_JOBSIZE * PPN)) --ppn $PPN --cpu-bind depth -d 16 ./local_rank.sh \
    tokenization.sh \
    --input-dir data \
    --output-dir data_Llama2Tokenizer \
    --num-workers 16 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /home/hzheng/AuroraGPT/datasets/llm_data/preprocess/tokenizer/tokenizer.model
```

## Tokenization Workflow

The above command launches the `tokenization.sh` script across multiple MPI ranks. Here's a breakdown:

1. **Environment Variables**  
   - `PBS_JOBSIZE`: Number of nodes allocated by PBS (unique hosts in your job).  
   - `PPN`: Processes per node (e.g., cores per node).  
   - Together, `-n $((PBS_JOBSIZE * PPN))` starts one MPI rank per process slot.

2. **MPI Rank Setup**  
   Each rank receives:  
   - `RANK` via `OMPI_COMM_WORLD_RANK` or `PMI_RANK`.  
   - `WORLD_SIZE` via `OMPI_COMM_WORLD_SIZE` or `PMI_SIZE`.  
   These are exported before `exec tokenization.sh`.

3. **Script Arguments**  
   - `--input-dir data`  
     Recursively finds all `*.gz` files under `data/`.  
   - `--output-dir data_Llama2Tokenizer`  
     Mirrors input directory structure under this path for tokenized outputs.  
   - `--num-workers 16`  
     Number of intra-file worker processes to use for each file.  
   - `--tokenizer-type Llama2Tokenizer`  
     Specifies the tokenizer backend; can be changed to `GPT2BPETokenizer`, etc.  
   - `--tokenizer-model /path/to/spiece.model`  
     Path to the SentencePiece model when required.

4. **Prerequisites**  
   - Ensure `mpi4py` and your tokenizer dependencies are installed in your environment.  
   - Make `tokenization.sh` executable:  
     ```bash
     chmod +x tokenization.sh
     ```
   - Adjust paths (`data/`, `data_Llama2Tokenizer`, model location) as needed.

5. **Running Manually**  
   For a quick local test without PBS, you can emulate two ranks:  
   ```bash
   OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_SIZE=2 \
   tokenization.sh --input-dir data --output-dir data_tok --num-workers 4 --tokenizer-type GPT2BPETokenizer
   ```