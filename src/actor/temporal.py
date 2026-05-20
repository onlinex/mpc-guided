"""Conv1d-equivalent temporal mixing as a single matmul.

Channels-last interface: ``(B, L, in_ch) -> (B, L, out_ch)``. Mathematically
identical to ``nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)``
applied to channels-first input with zero padding, but expressed as one
matmul over unfolded neighbour windows. That avoids the per-call
cuDNN/MKL setup overhead nn.Conv1d carries on small tensors — the kind we
hit in the actor decoder, where the chunk dim is single digits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv1d(nn.Module):
    """Same-padding 1D conv over the L (time) dim of a ``(B, L, in_ch)`` input.

    Weights are shared across positions (the conv inductive bias): the same
    kernel slides over neighbour windows, so neighbouring outputs are forced
    to share representation — a real temporal smoothness prior, unlike a
    per-position Linear.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        # Total pad = k-1 keeps output length == input length. For odd k it
        # splits evenly; for even k the right side gets one more pad slot
        # (output[t] sees t-(k//2-1)..t+(k//2) — a half-step forward bias).
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2
        # Weight layout (out_ch, in_ch * kernel_size) matches nn.Linear so
        # F.linear works directly, while the fan_in computation (dim 1)
        # matches nn.Conv1d's receptive-field convention.
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch * kernel_size))
        self.bias = nn.Parameter(torch.empty(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_ch * kernel_size)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad along L, then unfold a kernel-sized sliding window per position.
        if self.pad_left or self.pad_right:
            x = F.pad(x, (0, 0, self.pad_left, self.pad_right))
        # (B, L, in_ch, kernel_size) — the new last dim is the window.
        windows = x.unfold(1, self.kernel_size, 1)
        flat = windows.reshape(*windows.shape[:2], -1)  # (B, L, in_ch * kernel_size)
        return F.linear(flat, self.weight, self.bias)
