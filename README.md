
# LLM dataset utiltity
## Install
```bash
git clone https://github.com/llm_dataset.git
cd llm_dataset
pip install -e .
```
## Tokenizing datasat
```bash
mpiexec -n $((PBS_JOBSIZE * PPN)) --ppn $PPN --cpu-bind depth -d 16 launcher.sh \
    tokenization.sh \
    --input-dir data \
    --output-dir data_Llama2Tokenizer \
    --num-workers 16 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./llm_dataset/preprocess/tokenizer/tokenizer.model
```

**Script Arguments**  
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

## Todo
https://github.com/openai/tiktoken