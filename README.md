
# LLM dataset utiltity
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
{'dataset_idx': tensor([0, 0, 0, 0, 0, 0, 0, 0]), 'input_ids': tensor([[  550, 29871, 29896,  ..., 29871, 29906, 29900],
        [  393,  1369,  1196,  ...,   916, 27690,  4486],
        [29911,  4448,  3446,  ...,  2831,  1906,   451],
        ...,
        [ 1183,  4083,  5864,  ...,  3256,   304,  6568],
        [ 4644,  1336,  1632,  ...,  1730, 29892, 24438],
        [  714,   278,  7135,  ...,   313, 29924,  2965]]), 'labels': tensor([[  550, 29871, 29896,  ..., 29871, 29906, 29900],
        [  393,  1369,  1196,  ...,   916, 27690,  4486],
        [29911,  4448,  3446,  ...,  2831,  1906,   451],
        ...,
        [ 1183,  4083,  5864,  ...,  3256,   304,  6568],
        [ 4644,  1336,  1632,  ...,  1730, 29892, 24438],
        [  714,   278,  7135,  ...,   313, 29924,  2965]])}
```




https://github.com/openai/tiktoken
