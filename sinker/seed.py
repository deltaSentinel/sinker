"""Central seeding. Call once at every entrypoint; never scatter seeds."""
import random
import numpy as np
import torch

DEFAULT_SEED = 1337


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

