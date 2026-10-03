import numpy as np
import pytest

from conftest import SPOT_SLICE
from lesion_preservation.data import load_slices
from lesion_preservation.masks import apply_mask, random_mask
from lesion_preservation.recon import get_reconstructor


def reconstruct(s, speedup):
    mask = random_mask(s.kspace.shape[-1], speedup, 0.08, seed=0)
    return get_reconstructor("zero_filled")(apply_mask(s.kspace, mask), mask, s.reference.shape)


def test_full_mask_gives_reference(fake_scan):
    for s in load_slices(*fake_scan):
        np.testing.assert_array_equal(reconstruct(s, 1), s.reference)


def test_undersampled_differs(fake_scan):
    (s,) = load_slices(*fake_scan, slices=[SPOT_SLICE])
    image = reconstruct(s, 4)
    assert image.shape == s.reference.shape
    assert not np.allclose(image, s.reference)


def test_unknown_name_fails():
    with pytest.raises(KeyError, match="zero_filled"):
        get_reconstructor("varnet")
