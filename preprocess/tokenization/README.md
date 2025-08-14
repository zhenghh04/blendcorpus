# Tokenizing dataset
After install blendcorpus, one can use the tokenization script to tokenize the json raw files.
## Llama2Tokenizer
```bash
PPN=1
mpiexec -n $((PBS_JOBSIZE * PPN)) --ppn $PPN --cpu-bind depth -d 16 \
    tokenization \
    --input-dir data \
    --output-dir data_Llama2Tokenizer_eod \
    --num-workers 16 \
    --tokenizer-type Llama2Tokenizer \
    --append-eod \
    --tokenizer-model ./llm_dataset/preprocess/tokenizer/tokenizer.model
```

## Hugging Face tokenizer
In this case, one will have to provide the path to the tokenizer model

```bash
$ ls /home/hzheng/AuroraGPT/olmo-mix-1124/gemma-7b/
config.json             model-00001-of-00004.safetensors  model.safetensors.index.json  tokenizer.json
examples                model-00002-of-00004.safetensors  README.md                     tokenizer.model
gemma-7b.gguf           model-00003-of-00004.safetensors  special_tokens_map.json
generation_config.json  model-00004-of-00004.safetensors  tokenizer_config.json
```

```bash
export PPN=1
export NUM_WORKERS=64

export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export TOKENIZER=/home/hzheng/AuroraGPT/olmo-mix-1124/gemma-7b/
# input folder
export DATA=data_fused5
# output folder
export DATA_TOK=data_fused5_gemma_eod

export DS_ACCELERATOR=cpu
mpiexec -n $((PBS_JOBSIZE*PPN)) --ppn $PPN  --cpu-bind depth -d $NUM_WORKERS \
       launcher.sh tokenization \
       --input-dir $DATA \
       --output-dir $DATA_TOK \
       --num-workers $NUM_WORKERS \
       --tokenizer-type HFTokenizer \
       --append-eod \
       --tokenizer-model $TOKENIZER \
       --seq-length 10000000000000
```