import random

import numpy as np
import torch

from cxr.utils.seed import set_seed


def test_same_seed_same_draws():
    set_seed(3)
    a = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(3)
    b = (random.random(), np.random.rand(), torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


def test_disables_tf32_and_cudnn_autotune():
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(0)
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
