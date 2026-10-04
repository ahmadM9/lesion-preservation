import numpy as np
import pytest

from conftest import SPOT_SLICE
from lesion_preservation.data import Box, load_slices
from lesion_preservation.judge import get_judge
from lesion_preservation.masks import apply_mask, random_mask
from lesion_preservation.recon import get_reconstructor

RING_WIDTH = 2
THRESHOLD = 1.5


def judge():
    return get_judge("placeholder", ring_width=RING_WIDTH, threshold=THRESHOLD)


def reconstruct(s, speedup):
    mask = random_mask(s.kspace.shape[-1], speedup, 0.08, seed=0)
    return get_reconstructor("zero_filled")(apply_mask(s.kspace, mask), mask, s.reference.shape)


def test_planted_spot_found_at_1x(fake_scan):
    (s,) = load_slices(*fake_scan, slices=[SPOT_SLICE])
    (verdict,) = judge()(reconstruct(s, 1), s.boxes)
    assert verdict.box_id == s.boxes[0].box_id
    assert verdict.found


def test_same_box_without_spot_not_found(fake_scan):
    (spot,) = load_slices(*fake_scan, slices=[SPOT_SLICE])
    (empty,) = load_slices(*fake_scan, slices=[0])
    (verdict,) = judge()(reconstruct(empty, 1), spot.boxes)
    assert not verdict.found
    assert verdict.score == pytest.approx(1, abs=0.1)


def test_box_at_edge_clips_ring():
    image = np.ones((8, 8))
    image[:2, :2] = 3.0
    (verdict,) = judge()(image, [Box(0, 0, 0, 2, 2, "lesion")])
    assert verdict.score == pytest.approx(3.0)
    assert verdict.found


def test_one_verdict_per_box_in_order():
    image = np.ones((10, 10))
    boxes = [Box(7, 1, 1, 2, 2, "a"), Box(3, 6, 6, 2, 2, "b")]
    verdicts = judge()(image, boxes)
    assert [v.box_id for v in verdicts] == [7, 3]


def test_unknown_name_fails():
    with pytest.raises(KeyError, match="placeholder"):
        get_judge("nnunet")
