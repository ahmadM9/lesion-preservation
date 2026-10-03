import numpy as np
import pytest
from fastmri import evaluate
from skimage.metrics import structural_similarity

from lesion_preservation.data import Box
from lesion_preservation.metrics import box_scores, nmse, psnr, ssim

WIN = 7


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.5, 2.0, size=(20, 16))
    img = ref + 0.1 * rng.standard_normal(ref.shape)
    return ref, img


def test_slice_scores_match_fastmri(pair):
    ref, img = pair
    maxval = ref.max()
    assert psnr(ref, img, maxval) == pytest.approx(evaluate.psnr(ref, img, maxval))
    assert nmse(ref, img) == pytest.approx(evaluate.nmse(ref, img))
    mean, _ = ssim(ref, img, maxval, WIN)
    assert mean == pytest.approx(evaluate.ssim(ref[None], img[None], maxval)[0])


def test_box_psnr_uses_slice_scale(pair):
    ref, img = pair
    ref[0, 0] = 10.0  # slice maximum outside the box
    box = Box(0, 5, 4, 6, 5, "lesion")
    _, ssim_map = ssim(ref, img, ref.max(), WIN)
    scores = box_scores(ref, img, ssim_map, box, ref.max())
    ref_box, img_box = ref[5:11, 4:9], img[5:11, 4:9]
    assert scores["psnr"] == pytest.approx(psnr(ref_box, img_box, 10.0))
    assert scores["psnr"] != pytest.approx(psnr(ref_box, img_box, ref_box.max()))


def test_box_ssim_is_map_mean(pair):
    ref, img = pair
    box = Box(0, 2, 3, 4, 3, "lesion")
    _, ssim_map = ssim(ref, img, ref.max(), WIN)
    _, expected = structural_similarity(ref, img, data_range=ref.max(), win_size=WIN, full=True)
    scores = box_scores(ref, img, ssim_map, box, ref.max())
    assert scores["ssim"] == pytest.approx(expected[2:6, 3:6].mean())


def test_identical_images():
    ref = np.random.default_rng(1).uniform(size=(12, 12))
    mean, ssim_map = ssim(ref, ref, ref.max(), WIN)
    scores = box_scores(ref, ref, ssim_map, Box(0, 1, 1, 3, 4, "lesion"), ref.max())
    assert psnr(ref, ref, ref.max()) == np.inf and scores["psnr"] == np.inf
    assert mean == pytest.approx(1.0) and scores["ssim"] == pytest.approx(1.0)
    assert nmse(ref, ref) == 0.0 and scores["nmse"] == 0.0
