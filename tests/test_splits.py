import pytest
import torch

from graphguard.splits import make_temporal_split


def test_temporal_split_is_non_overlapping_and_excludes_unknowns():
    time_step = torch.tensor([1, 10, 29, 30, 34, 35, 49, 49])
    labels = torch.tensor([0, 1, -1, 0, 1, 0, -1, 1])

    split = make_temporal_split(
        time_step,
        labels,
        train_end=29,
        validation_start=30,
        validation_end=34,
        test_start=35,
    )

    assert split.train_mask.tolist() == [True, True, False, False, False, False, False, False]
    assert split.validation_mask.tolist() == [False, False, False, True, True, False, False, False]
    assert split.test_mask.tolist() == [False, False, False, False, False, True, False, True]
    assert not bool((split.train_mask & split.validation_mask).any())
    assert not bool((split.train_mask & split.test_mask).any())
    assert not bool((split.validation_mask & split.test_mask).any())


def test_temporal_split_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        make_temporal_split(
            torch.tensor([1, 2]),
            torch.tensor([0]),
            train_end=1,
            validation_start=2,
            validation_end=2,
            test_start=3,
        )


def test_temporal_split_rejects_bad_boundaries():
    with pytest.raises(ValueError, match="Temporal boundaries"):
        make_temporal_split(
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 0, 1]),
            train_end=2,
            validation_start=2,
            validation_end=2,
            test_start=3,
        )
