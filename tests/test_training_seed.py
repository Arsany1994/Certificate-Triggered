import random

import numpy as np
import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import seed_everything


def sample(seed):
    seed_everything(seed)
    return (
        random.random(),
        np.random.rand(4),
        torch.rand(4),
    )


def main():
    assert Parameters(seed=0).seed == 0
    assert Parameters(seed=2).seed == 2

    try:
        Parameters(seed=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative training seed was accepted")

    first = sample(123)
    second = sample(123)

    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])

    other = sample(124)

    assert first[0] != other[0]
    assert not np.array_equal(first[1], other[1])
    assert not torch.equal(first[2], other[2])

    print("PASS: configurable training seed validation")
    print("PASS: repeated seed produces identical Python/NumPy/Torch streams")
    print("PASS: different seeds produce different streams")


if __name__ == "__main__":
    main()
