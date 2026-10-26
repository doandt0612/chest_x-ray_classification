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
