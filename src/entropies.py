from typing import List
import torch
import numpy as np


# https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
# https://de.wikipedia.org/wiki/Differentielle_Entropie
NORMALFACTOR = (0.5 * np.log(2 * np.pi * np.e))


def rel_entropy(first: torch.tensor, sec: torch.tensor, off: float = 1e-12) -> torch.tensor:
    """
    Calculate the relative entropy over two batches of activations
    :param first: torch.tensor = [batch, 1D activation]
    :param sec: torch.tensor = [batch, 1D activation]
    :param off: float = offset to apply to prevent zeros in first and sec
    :return: torch.tensor = [batch] relative entropy for each sample in batch
    """

    # ensure positive definiteness of tensors
    first -= torch.min(first, dim=1).values[:, None]
    sec -= torch.min(sec, dim=1).values[:, None]

    # prevent zero activation
    first += off
    sec += off

    # normalize
    first /= torch.sum(first, dim=1)[:, None]
    sec /= torch.sum(sec, dim=1)[:, None]

    # calculate relative entropy for x > 0, y > 0
    return torch.sum(first * torch.log(first / sec), dim=1)


def diff_entropy(actis: torch.tensor) -> torch.tensor:
    """
    Calculate the differential entropy between a set of activations
    :param actis: torch.tensor = [batch, activations]
    :return: float = differential entropy
    """
    # normalize
    actis /= torch.sum(actis, dim=1)[:, None]

    # get std
    std = torch.std(actis, dim=0)

    # calculate entropy
    return torch.mean(NORMALFACTOR + torch.log(std))


def cutoff_det(seq: torch.tensor, atol: float = 1e-8, rtol: float=1e-5) -> torch.tensor:
    """
    Search for saturation if seq with tolerance
    :param seq: torch.tensor = [batch, 1D entropy]
    :param atol: float = absolut tolerance when comparing floats
    :param rtol: float = relative tolerance when comparing floats
    :return: torch.tensor = [batch] cutoffs for each run in batch
    """
    cutoffs = torch.full((len(seq),), len(seq[0]), dtype=torch.int32, requires_grad=False)
    for i, entropies in enumerate(seq):
        for j, entropy in enumerate(entropies[::-1]):
            if not (np.isclose(entropy, entropies[-1], atol=atol, rtol=rtol)):
                cutoffs[i] = len(entropies) - j - 1
                break
    return cutoffs