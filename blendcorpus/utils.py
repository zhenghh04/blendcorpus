# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""
from functools import reduce
import math
import operator

import torch
from typing import Optional
import logging
import os
try:
    rank = int(os.environ['RANK'])
except:
    rank = 0

_DLIO_PROFILER_EXIST = True
_DFTRACER_EXIST = True

try:
    import dftracer  # type:ignore
except Exception:
    _DFTRACER_EXIST = False

try:
    import dlio_profiler  # type:ignore
except Exception:
    _DLIO_PROFILER_EXIST = False

if _DFTRACER_EXIST:
    from dftracer.logger import (  # type:ignore
        dftracer as PerfTrace,
        dft_fn as Profile,
        DFTRACER_ENABLE as DFTRACER_ENABLE,
    )

elif _DLIO_PROFILER_EXIST:
    from dlio_profiler.logger import fn_interceptor as Profile  # type:ignore
    from dlio_profiler.logger import dlio_logger as PerfTrace  # type:ignore
else:
    from functools import wraps

    class Profile(object):
        def __init__(
            self, cat, name=None, epoch=None, step=None, image_idx=None, image_size=None
        ):
            return

        def log(self, func):
            return func

        def log_init(self, func):
            return func

        def iter(self, func, iter_name="step"):
            return func

        def __enter__(self):
            return

        def __exit__(self, type, value, traceback):
            return

        def update(
            self, epoch=None, step=None, image_idx=None, image_size=None, args={}
        ):
            return

        def flush(self):
            return

        def reset(self):
            return

        def log_static(self, func):
            return

    class dftracer(object):
        def __init__(
            self,
        ):
            self.type = None

        def initialize_log(self, logfile=None, data_dir=None, process_id=-1):
            return

        def get_time(self):
            return

        def enter_event(self):
            return

        def exit_event(self):
            return

        def log_event(self, name, cat, start_time, duration, string_args=None):
            return

        def finalize(self):
            return

    PerfTrace = dftracer()
    DFTRACER_ENABLE = False

def get_logger(
    name: str,
    level: Optional[str] = None,
    rank_zero_only: Optional[bool] = True,
) -> logging.Logger:
    """Returns a `logging.Logger` object.

    If `rank_zero_only` passed, the level will be set to CRITICAL on all
    non-zero ranks (and will be set to `level` on RANK==0).
    """
    logger = logging.getLogger(name)
    logger.setLevel(
        str(level if level is not None else os.environ.get("LOG_LEVEL", "INFO")).upper()
    )
    if rank_zero_only and rank != 0:
        logger.setLevel("CRITICAL")
    return logger

def current_device_name():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return torch.cuda.get_device_name(idx)
    elif torch.xpu.is_available():
        idx = torch.xpu.current_device()
        return torch.xpu.get_device_name(idx)
    else:
        return "cpu"

class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = \
                torch.empty(required_len,
                            dtype=dtype,
                            device=current_device_name(),
                            requires_grad=False)

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def get_attr_wrapped_model(model, attr, allow_none=True):
    """Get an attribute from a wrapped model"""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:
        def condition(model, attr):
            return not hasattr(model, attr)
    else:
        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module
    return getattr(model, attr)


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty(
        (1,),
        dtype = inp.dtype,
        device = inp.device,
        requires_grad = requires_grad,
    )
    out.data = inp.data
    return out

class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''
    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)

def assert_viewless_tensor(tensor, extra_msg = None):
    '''Assert that a tensor is not a view (i.e., its '._base' field is
    not set).'''
    if isinstance(tensor, list):
        [ assert_viewless_tensor(t) for t in tensor ]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor

def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    '''
    assert_viewless_tensor(tensor, extra_msg = "FYI, tensor._base has shape %s, and new_data_tensor has shape %s." % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape))
    tensor.data = new_data_tensor

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_
import datetime
def print_rank_0(msg):
    if rank==0:
        print(f" [INFO][{datetime.datetime.now()}] {msg}", flush=True)



def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
    skip_mask=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None
    if not skip_mask:
        attention_mask = torch.tril(
            torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):
            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if (
                    reset_attention_mask
                    and not skip_mask
                    and attention_mask is not None
                ):
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    if not skip_mask:
        assert attention_mask is not None
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
