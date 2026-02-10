# LLM Data Pipeline

<p align="center">
  <img src="./.docs/figures/BlendCorpus.jpg" alt="BlendCorpus Logo" width="400"/>
</p>

**BlendCorpus** is a modular and scalable data preprocessing and loading framework for large language model (LLM) training. It supports efficient tokenization using MPI-based parallelism and provides customizable dataloaders compatible with various LLM training workflows such as Megatron-DeepSpeed and TorchTitan. **BlendCorpus** allows seamless integration of different tokenizers, dataset formats, and distributed training setups, making it suitable for research- and production-scale LLM pipelines.

## Install
```bash
git clone https://github.com/blendcorpus.git
cd blendcorpus
pip install -e .
```

## Downloading a Dataset from Hugging Face
Set the `HF_TOKEN`. If you do not have a Hugging Face token, you can generate one: https://huggingface.co/settings/tokens
```bash
export HF_TOKEN=xxxxxxxx
download-huggingface-dataset.sh --dataset HuggingFaceFW/fineweb-edu --output fineweb-edu-2025-09-05
```

## Fusing Small Files
Fusing small files reduces index-building overhead in Megatron-DeepSpeed and helps avoid rounding errors that can occur when processing very small files. By combining these files, you improve efficiency and stability during training or data preprocessing.

To estimate how many fused files will be produced without writing outputs, use `--dry-run`.
Single-process example:
```bash
preprocess/fuse_files/fuse_files_parallel.sh --input-dir data --dry-run
```
Once we figure out how many files it will generate, we can choose NPROCS that is close to that files. 

```bash
export PPN=4
export THREADS_PER_RANK=16
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN --cpu-bind depth -d $THREADS_PER_RANK launcher.sh \
  fuse_files_parallel.sh --input-dir data --output-dir data-fused
```

## Tokenizing the Dataset
```bash
mpiexec -n $((PBS_JOBSIZE * PPN)) --ppn $PPN --cpu-bind depth -d 16 \
    tokenization \
    --input-dir data-fused \
    --output-dir data-fused-tok \
    --num-workers 16 \
    --tokenizer-type Llama2Tokenizer \
    --append-eod \
    --tokenizer-model ./llm_dataset/preprocess/tokenizer/tokenizer.model
```

This creates tokenized data in the `data-fused-tok` folder. Settings are logged in the `data_Llama2Tokenizer_eod` folder.

**Script arguments**
   - `--input-dir data`
     Recursively finds all `*.gz` files under `data/`.
   - `--output-dir data_Llama2Tokenizer`
     Mirrors the input directory structure under this path for tokenized outputs.
   - `--num-workers 16`
     Number of intra-file worker processes to use for each file.
   - `--tokenizer-type Llama2Tokenizer`
     Specifies the tokenizer backend; can be changed to `GPT2BPETokenizer`, etc.
   - `--tokenizer-model /path/to/spiece.model`
     Path to the SentencePiece model when required.
   - `--append-eod`
     Append the end-of-document token.

## Generate Dataset Metadata

```bash
mpiexec -np $NPROCS --ppn $PPN --cpu-bind depth -d 1 launcher.sh \
	      get_meta_data \
        --input-dir data-fused-tok \
        --output data-fused-tok.json
```
`data-fused-tok.json` contains `num_docs` and `num_tokens` for all files.


## Generate Data List File
```bash
gen_file_list --input-json data-fused-tok.json 
  --output olmo-fused-file-list.txt --topdir /flare/AuroraGPT/datasets/olmo-mix-1124/  --epochs 1 2 3
```
You can specify the epochs for different corpora.

## Testing the Dataset with `tests/`
Use `tests/test_dataloader.py` to validate dataset construction and dataloader iteration with your generated file list.

```bash
mpiexec -n 1 python tests/test_dataloader.py \
  --trace-dir ./trace \
  --data-file-list olmo-fused-file-list.txt \
  --global-batch-size 8 \
  --train-iters 10 \
  --seq-length 4096 \
  --micro-batch-size 8 \
  --num-workers 2 \
  --dataloader-iter
```

Notes:
- `--data-file-list` should point to the file produced by `gen_file_list`.
- Increase `-n` for multi-rank testing when MPI is configured.
- Set `--print-sample-info` if you also want per-sample corpus/prefix debug output.

## Known Issues
* Incomplete dataset download
* Incomplete file fusion - resubmit the jobs
* Incomplete tokenization - resubmit the jobs

## Using the Dataset and Dataloader
```python
from blendcorpus import (
    get_config, 
    set_config, 
    mpu, 
    build_gpt_datasets, 
    build_pretraining_data_loader
)

from blendcorpus.utils import get_ltor_masks_and_position_ids

mpu.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel_size=1,
    )

# Check `blendcorpus.data.config` for details.
set_config(args)
config = get_config()

# Build datasets
train_ds, valid_ds, test_ds = build_gpt_datasets(config)

# Build dataloaders
# consumed_train_samples = restart_iters * args.global_batch_size
train_dataloader = build_pretraining_data_loader(
        train_ds, 
        consumed_train_samples, 
        config)
valid_dataloader = build_pretraining_data_loader(
        valid_ds, consumed_valid_samples, config)
test_dataloader = build_pretraining_data_loader(
        test_ds, consumed_test_samples, config)

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

    # Items and their types.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data in model parallel group
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    
    data_b = mpu.broadcast_data_in_model_parallel_group(keys, data, datatype)

    # Unpack
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position IDs.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

```

Each item from the dataloader has size `args.micro_batch_size` (8 in this example):
```
 {'dataset_idx': tensor([0, 0, 0, 0, 0, 0, 0, 0]), 'text': tensor([[29091,   350,  1525,  ..., 29890,   440,   487],
        [29891,  6378, 29889,  ...,   448, 18043,   491],
        [  508,  5142,  8037,  ...,   393,  2367,   963],
        ...,
        [  261,  5866, 18296,  ...,   322,  8820,   526],
        [  459,   793, 29915,  ...,  1735, 29889,    13],
        [29915, 29879,  4315,  ...,  6115, 24060, 18864]])}
```
