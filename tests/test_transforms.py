import fastmri
import numpy as np
import pytest
import torch
from fastmri.data.transforms import center_crop as fm_center_crop
from fastmri.data.transforms import to_tensor

from lesion_preservation.transforms import center_crop, fft2c, ifft2c, rss

SHAPES = [(4, 16, 12), (3, 15, 13)]


def complex_coils(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def to_numpy(tensor):
    return torch.view_as_complex(tensor.contiguous()).numpy()


def assert_close(actual, expected):
    np.testing.assert_allclose(
        actual, expected, rtol=1e-5, atol=1e-6 * float(np.abs(expected).max())
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_fft2c_matches_fastmri(shape):
    image = complex_coils(shape, seed=0)
    assert_close(fft2c(image), to_numpy(fastmri.fft2c(to_tensor(image))))


@pytest.mark.parametrize("shape", SHAPES)
def test_ifft2c_matches_fastmri(shape):
    kspace = complex_coils(shape, seed=1)
    assert_close(ifft2c(kspace), to_numpy(fastmri.ifft2c(to_tensor(kspace))))


@pytest.mark.parametrize("shape", SHAPES)
def test_round_trip(shape):
    image = complex_coils(shape, seed=2)
    assert_close(ifft2c(fft2c(image)), image)


@pytest.mark.parametrize("shape", SHAPES)
def test_rss_matches_fastmri(shape):
    coil_images = complex_coils(shape, seed=3)
    expected = fastmri.rss_complex(to_tensor(coil_images), dim=0).numpy()
    assert_close(rss(coil_images, coil_axis=0), expected)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("crop", [(8, 8), (7, 9), (11, 6)])
def test_center_crop_matches_fastmri(shape, crop):
    image = np.abs(complex_coils(shape, seed=4))
    expected = fm_center_crop(torch.from_numpy(image), crop).numpy()
    np.testing.assert_array_equal(center_crop(image, crop), expected)


def test_center_crop_too_large():
    with pytest.raises(ValueError):
        center_crop(np.zeros((8, 8)), (9, 8))
