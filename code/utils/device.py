"""PyTorch device selection: prefer Apple MPS, then CUDA, fallback CPU."""

from __future__ import annotations


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
