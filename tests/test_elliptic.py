import pytest

from graphguard.elliptic import RAW_CLASS_MAP


def test_raw_elliptic_class_semantics_are_explicit():
    assert RAW_CLASS_MAP == {"unknown": -1, "1": 1, "2": 0}


def test_unexpected_raw_class_is_not_silently_accepted():
    with pytest.raises(KeyError):
        _ = RAW_CLASS_MAP["unexpected"]
