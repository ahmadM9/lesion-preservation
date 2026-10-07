import numpy as np

from conftest import SPOT_SLICE
from lesion_preservation.data import Box, load_slices, with_neighbours
from lesion_preservation.normal import brain_mask, place_normal_boxes

CLEARANCE = 2
MAX_DARK = 0.25


def spot_slice(fake_scan):
    (s,) = load_slices(*fake_scan, slices=[SPOT_SLICE])
    return s


def footprint(box, shape, grow=0):
    inside = np.zeros(shape, dtype=bool)
    inside[
        max(box.row - grow, 0) : box.row + box.height + grow,
        max(box.col - grow, 0) : box.col + box.width + grow,
    ] = True
    return inside


def test_brain_mask_leaves_out_background(fake_scan):
    s = spot_slice(fake_scan)
    mask = brain_mask(s.reference, 0)
    # the fake head ends one pixel inside the crop
    assert not mask[0].any() and not mask[-1].any()
    assert mask[1:-1, 1:-1].all()
    assert brain_mask(s.reference, 1).sum() < mask.sum()


def test_same_size_inside_brain_clear_of_lesion(fake_scan):
    s = spot_slice(fake_scan)
    (lesion,) = s.boxes
    (normal,) = place_normal_boxes(s.reference, s.boxes, s.boxes, 0, 0, CLEARANCE, MAX_DARK, [])
    assert (normal.box_id, normal.height, normal.width) == (
        lesion.box_id, lesion.height, lesion.width,
    )  # fmt: skip
    grown = footprint(normal, s.reference.shape, CLEARANCE)
    assert not (grown & ~brain_mask(s.reference, 0)).any()
    assert not (grown & footprint(lesion, s.reference.shape)).any()


def test_same_place_every_call(fake_scan):
    s = spot_slice(fake_scan)
    first = place_normal_boxes(s.reference, s.boxes, s.boxes, 0, 0, CLEARANCE, MAX_DARK, [])
    assert place_normal_boxes(s.reference, s.boxes, s.boxes, 0, 0, CLEARANCE, MAX_DARK, []) == first


def test_clear_of_other_labels_and_of_each_other():
    reference = np.zeros((40, 40))
    reference[2:-2, 2:-2] = 1.0
    lesions = [Box(0, 5, 5, 4, 4, "lesion"), Box(1, 5, 25, 4, 4, "lesion")]
    other = Box(2, 25, 5, 10, 10, "other")
    normals = place_normal_boxes(
        reference, [*lesions, other], lesions, 0, 0, CLEARANCE, MAX_DARK, []
    )
    assert len(normals) == 2
    taken = [footprint(b, reference.shape) for b in [*lesions, other]]
    for normal in normals:
        grown = footprint(normal, reference.shape, CLEARANCE)
        assert not any((grown & t).any() for t in taken)
        taken.append(footprint(normal, reference.shape))


def test_no_room_gives_no_box(fake_scan):
    s = spot_slice(fake_scan)
    assert place_normal_boxes(s.reference, s.boxes, s.boxes, 0, 7, CLEARANCE, MAX_DARK, []) == []


def dark_fraction(reference, box):
    inside = reference[box.row : box.row + box.height, box.col : box.col + box.width]
    return (inside == 0).mean()


def test_dark_hole_inside_brain_avoided():
    # a fluid-filled hole: inside the brain mask, since holes are filled, but not tissue
    reference = np.zeros((40, 40))
    reference[2:-2, 2:-2] = 1.0
    reference[10:30, 10:30] = 0.0
    lesion = Box(0, 4, 4, 4, 4, "lesion")
    darkest = 0.0
    for seed in range(20):
        (normal,) = place_normal_boxes(
            reference, [lesion], [lesion], seed, 0, CLEARANCE, MAX_DARK, []
        )
        assert dark_fraction(reference, normal) <= MAX_DARK
        (anywhere,) = place_normal_boxes(reference, [lesion], [lesion], seed, 0, CLEARANCE, 1.0, [])
        darkest = max(darkest, dark_fraction(reference, anywhere))
    # without the rule some boxes land in the hole
    assert darkest > MAX_DARK


def test_fluid_on_neighbouring_slice_avoided():
    # this slice is all tissue; the slice above has a dark hole, as a thick slice grazing a
    # ventricle would show it grey here and dark there
    reference = np.zeros((40, 40))
    reference[2:-2, 2:-2] = 1.0
    above = reference.copy()
    above[10:30, 10:30] = 0.0
    lesion = Box(0, 4, 4, 4, 4, "lesion")
    for seed in range(20):
        (normal,) = place_normal_boxes(
            reference, [lesion], [lesion], seed, 0, CLEARANCE, MAX_DARK, [above]
        )
        assert dark_fraction(above, normal) <= MAX_DARK


def test_with_neighbours(fake_scan):
    pairs = list(with_neighbours(load_slices(*fake_scan), 1))
    assert [s.index for s, _ in pairs] == [0, 1, 2]
    assert [len(near) for _, near in pairs] == [1, 2, 1]
    middle, near = pairs[1]
    assert np.array_equal(near[0], pairs[0][0].reference)
    assert [len(near) for _, near in with_neighbours(load_slices(*fake_scan), 0)] == [0, 0, 0]
