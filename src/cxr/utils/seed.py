"""Fix random seeds so a training run can be repeated."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all GPUs); make cuDNN pick deterministic kernels.

    GPU results can still differ in the last digits across hardware or library versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    use_full_precision()


def use_full_precision() -> None:
    """Turn off TF32, which is on by default for cuDNN convolutions on Ampere+ GPUs.

    TF32 shifts probabilities by ~1e-3 versus CPU, enough to flip decisions near the
    threshold; training, evaluation and the CPU API must all compute in full float32.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
