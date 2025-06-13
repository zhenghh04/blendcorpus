from . import parallel_state as mpu

# Add dataclass and typing imports
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    aml_data_download_path: Optional[str] = None
    data_path: Optional[List[str]] = None
    data_file_list: Optional[str] = None
    shuffle_sample_in_corpus: bool = False
    blend_sample_in_corpus: bool = False
    split: str = '969, 30, 1'
    train_data_path: Optional[List[str]] = None
    valid_data_path: Optional[List[str]] = None
    test_data_path: Optional[List[str]] = None
    data_cache_path: Optional[str] = None
    vocab_size: Optional[int] = None
    vocab_file: Optional[str] = None
    merge_file: Optional[str] = None
    vocab_extra_ids: int = 0
    seq_length: Optional[int] = None
    encoder_seq_length: Optional[int] = None
    decoder_seq_length: Optional[int] = None
    retriever_seq_length: int = 256
    sample_rate: float = 1.0
    mask_prob: float = 0.15
    short_seq_prob: float = 0.1
    mmap_warmup: bool = False
    num_workers: int = 2
    tokenizer_type: Optional[str] = None
    tokenizer_model: Optional[str] = None
    trust_remote_code: bool = False
    data_impl: str = 'infer'
    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    eod_mask_loss: bool = False
    train_data_exact_num_epochs: Optional[int] = None
    return_data_index: bool = False
    data_efficiency_curriculum_learning: bool = False
    train_idx_path: Optional[str] = None
    train_desc_path: Optional[str] = None
    train_doc_idx_path: Optional[str] = None
    train_sample_idx_path: Optional[str] = None
    train_shuffle_idx_path: Optional[str] = None
    repeated_dataloader: bool = False
    multiprocessing_context: str = 'fork'

# Set DATA_CONFIG as an instance of DataConfig
DATA_CONFIG = DataConfig()

def get_args():
    return DATA_CONFIG

def get_config():
    return DATA_CONFIG

def set_config(config):
    """
    Override DATA_CONFIG fields using an external config.
    `config` can be a dict of key/value pairs or an object with attributes.
    """
    # Determine mapping of overridden values
    if isinstance(config, dict):
        items = config.items()
    else:
        # Use object's __dict__
        items = vars(config).items()
    # Apply overrides
    for key, val in items:
        if hasattr(DATA_CONFIG, key):
            setattr(DATA_CONFIG, key, val)

from .utils import print_rank_0