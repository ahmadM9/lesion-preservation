import numpy as np
import pytest
import torch
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data.transforms import apply_mask as fm_apply_mask
from fastmri.data.transforms import to_tensor

from lesion_preservation.masks import apply_mask, effective_speedup, random_mask


def fastmri_mask(num_cols, speedup, center_fraction, seed):
    mask, _ = RandomMaskFunc([center_fraction], [speedup])((1, 8, num_cols, 2), seed=seed)
    return mask.numpy().reshape(num_cols).astype(bool)


@pytest.mark.parametrize("num_cols", [320, 396, 13])
@pytest.mark.parametrize("speedup, center_fraction", [(2, 0.08), (4, 0.08), (8, 0.04)])
@pytest.mark.parametrize("seed", [0, 1234, (7, 8, 9)])
def test_matches_fastmri(num_cols, speedup, center_fraction, seed):
    expected = fastmri_mask(num_cols, speedup, center_fraction, seed)
    np.testing.assert_array_equal(random_mask(num_cols, speedup, center_fraction, seed), expected)


def test_centre_rounds_half_to_even():
    # 10 * 0.25 = 2.5 rounds to 2
    mask = random_mask(10, 1e9, 0.25, seed=0)
    np.testing.assert_array_equal(np.flatnonzero(mask), [4, 5])
    np.testing.assert_array_equal(mask, fastmri_mask(10, 1e9, 0.25, seed=0))


def test_non_integer_speedup_matches_fastmri():
    expected = fastmri_mask(320, 2.5, 0.08, seed=5)
    np.testing.assert_array_equal(random_mask(320, 2.5, 0.08, seed=5), expected)


def test_speedup_one_keeps_all_lines():
    assert random_mask(320, 1, 0.08, seed=0).all()


def test_seed_decides_mask():
    first = random_mask(320, 4, 0.08, seed=3)
    np.testing.assert_array_equal(first, random_mask(320, 4, 0.08, seed=3))
    assert not np.array_equal(first, random_mask(320, 4, 0.08, seed=4))


def test_apply_mask_matches_fastmri():
    rng = np.random.default_rng(0)
    shape = (4, 16, 40)
    kspace = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    expected, _, _ = fm_apply_mask(to_tensor(kspace), RandomMaskFunc([0.08], [4]), seed=11)
    actual = apply_mask(kspace, random_mask(40, 4, 0.08, seed=11))
    np.testing.assert_array_equal(actual, torch.view_as_complex(expected).numpy())


def test_effective_speedup():
    assert effective_speedup(np.array([True, False, False, True, False, False])) == 3.0
