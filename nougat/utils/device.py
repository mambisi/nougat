"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import logging
try:
    import torch_xla.core.xla_model as xm
    has_xla = True
except ImportError:
    has_xla = False

def default_batch_size():
    if torch.cuda.is_available():
        batch_size = int(
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
        )
        if batch_size == 0:
            logging.warning("GPU VRAM is too small. Computing on CPU.")
    elif has_xla and xm.xla_device():
        # Adjusting for a TPU with 2.8GiB of memory per core as an example
        memory_per_core_gb = 2.8  # Specific to your TPU configuration
        # Assuming a similar 0.3 ratio of total memory usage as with GPUs, adjust as necessary
        batch_size = int(memory_per_core_gb * 1000 / 1024 * 0.3)  # Adjusted heuristic for 2.8GiB/core
    elif torch.backends.mps.is_available():
        # Heuristically choosing bs=4 for MPS
        batch_size = 4
    else:
        # Fallback for CPU or unknown devices
        batch_size = 1
        logging.warning("No GPU or TPU found. Conversion on CPU is very slow.")
    return batch_size

def move_to_device(model, bf16: bool = True, cuda: bool = True, xla: bool = True):
    try:
        if torch.backends.mps.is_available():
            return model.to("mps")
    except AttributeError:
        pass
    if bf16 and not xla:  # XLA does not support bfloat16 via .to() directly
        model = model.to(torch.bfloat16)
    if xla and has_xla:
        model = model.to(xm.xla_device())
    elif cuda and torch.cuda.is_available():
        model = model.to("cuda")
    return model