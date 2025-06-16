# LLM data pipeline

<p align="center">
  <img src="./.docs/figures/BlendCorpus.jpg" alt="BlendCorpus Logo" width="200"/>
</p>

**BlendCorpus** is a modular and scalable data preprocessing and loading framework for large language model (LLM) training. It supports efficient tokenization using MPI-based parallelism and provides customizable dataloaders compatible with various LLM training workflows such as Megatron-DeepSpeed and TorchTitan. **BlendCorpus** allows seamless integration of different tokenizers, dataset formats, and distributed training setups, making it suitable for research and production-scale LLM pipelines.

## Install
```bash
git clone https://github.com/blendcorpus.git
cd blendcorpus
pip install -e .
```
## Tokenizing datasat
```bash
mpiexec -n $((PBS_JOBSIZE * PPN)) --ppn $PPN --cpu-bind depth -d 16 launcher.sh \
    tokenization.sh \
    --input-dir data \
    --output-dir data_Llama2Tokenizer_eod \
    --num-workers 16 \
    --tokenizer-type Llama2Tokenizer \
    --append-eod \
    --tokenizer-model ./llm_dataset/preprocess/tokenizer/tokenizer.model
```

This will create tokenized data in ``data_Llama2Tokenizer_eod`` folder. Settings will be logged in ``data_Llama2Tokenizer_eod`` folder. 

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
   - `--append-eod` 
     Whether to append end of document token or not. 

## Using the dataset and dataloader
```python

from blendcorpus import (
    get_config, 
    set_config, 
    mpu, 
    build_gpt_datasets, 
    build_pretraining_data_loader
)

mpu.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel_size=1,
    )

# check blendcorpus.data.config for details 
set_config(args)
config = get_config()

# build datasets
train_ds, valid_ds, test_ds = build_gpt_datasets(config)

# build dataloaders
# consumed_train_samples = restart_iters * args.global_batch_size
train_dataloader = build_pretraining_data_loader(
        train_ds, consumed_train_samples, config)
valid_dataloader = build_pretraining_data_loader(
        valid_ds, consumed_valid_samples, config)
test_dataloader = build_pretraining_data_loader(test_ds, consumed_test_samples, config)


    # Build iterators.
dl_type = args.dataloader_type
assert dl_type in ["single", "cyclic"]

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

if train_dataloader is not None:
  train_data_iterator = (
      iter(train_dataloader)
        if dl_type == "single"
        else iter(cyclic_iter(train_dataloader))
    )

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


```

Each item from the data loader is (size of args.micro_batch_size (8 in this case))
```
 {'dataset_idx': tensor([0, 0, 0, 0, 0, 0, 0, 0]), 'text': tensor([[29091,   350,  1525,  ..., 29890,   440,   487],
        [29891,  6378, 29889,  ...,   448, 18043,   491],
        [  508,  5142,  8037,  ...,   393,  2367,   963],
        ...,
        [  261,  5866, 18296,  ...,   322,  8820,   526],
        [  459,   793, 29915,  ...,  1735, 29889,    13],
        [29915, 29879,  4315,  ...,  6115, 24060, 18864]])}
```




https://github.com/openai/tiktoken
