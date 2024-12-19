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
    actis = actis.double()
    # actis += 1
    actis /= torch.sum(actis, dim=1)[:, None]

    # get std
    logs = torch.log(torch.std(actis, dim=0))
    # logs = logs[np.where(np.logical_not(np.isinf(logs)))]

    # calculate entropy
    return torch.mean(NORMALFACTOR + logs)


def cutoff_det(seq: torch.tensor, atol: float = 1e-8, rtol: float = 1e-5, method: str = "absolut") -> torch.tensor:
    """
    Search for saturation if seq with tolerance
    :param seq: torch.tensor = [batch, 1D entropy]
    :param atol: float = absolut tolerance when comparing floats
    :param rtol: float = relative tolerance when comparing floats
    :param method: str = method used to determine cutoff [absolut, differential]
    :return: torch.tensor = [batch] cutoffs for each run in batch
    """
    if method == "differential":
        gaussian = np.exp(-(np.linspace(-10, 10, 5) / 0.1) ** 2 / 2)
        #import matplotlib.pyplot as plt

        #plt.plot(seq[0])
        seq = [- np.diff(np.convolve(entropies, gaussian, mode="valid")) for entropies in seq]
        #plt.plot(seq[0])
        #plt.show()

    elif method != "absolut":
        raise ValueError(f"Method {method} unknown")

    cutoffs = torch.full((len(seq),), len(seq[0]), dtype=torch.int32, requires_grad=False)
    for i, entropies in enumerate(seq):
        reference = entropies[-1] if method == "absolut" else 0
        for j, entropy in enumerate(entropies[::-1]):
            if not (np.isclose(entropy, reference, atol=atol, rtol=rtol)):
                cutoffs[i] = len(entropies) - j - 1
                break
    return cutoffs
