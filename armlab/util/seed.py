import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python, numpy and torch. Returns a numpy Generator for new code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
