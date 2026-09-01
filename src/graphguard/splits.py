from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological masks for labeled nodes."""

    train_mask: Tensor
    validation_mask: Tensor
    test_mask: Tensor
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int


def make_temporal_split(
    time_step: Tensor,
    labels: Tensor,
    *,
    train_end: int,
    validation_start: int,
    validation_end: int,
    test_start: int,
    unknown_label: int = -1,
) -> TemporalSplit:
    """Build non-overlapping chronological masks using only labeled nodes.

    Boundaries are inclusive. For example, train_end=29 and validation_start=30
    places timesteps 1-29 in training and 30 onward into later partitions.
    """
    if time_step.ndim != 1 or labels.ndim != 1:
        raise ValueError("time_step and labels must be 1-D tensors")
    if time_step.numel() != labels.numel():
        raise ValueError("time_step and labels must have the same length")
    if not (
        train_end < validation_start <= validation_end < test_start
    ):
        raise ValueError("Temporal boundaries must satisfy train < validation <= validation < test")

    labeled = labels != unknown_label
    train_mask = (time_step <= train_end) & labeled
    validation_mask = (
        (time_step >= validation_start)
        & (time_step <= validation_end)
        & labeled
    )
    test_mask = (time_step >= test_start) & labeled

    if bool((train_mask & validation_mask).any()) or bool((train_mask & test_mask).any()) or bool((validation_mask & test_mask).any()):
        raise RuntimeError("Temporal split masks overlap")

    return TemporalSplit(
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
    )
