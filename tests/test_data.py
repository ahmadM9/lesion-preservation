import h5py
import numpy as np
import pandas as pd
from fastmri.data.mri_data import AnnotatedSliceDataset

from conftest import FAKE_CROP, FAKE_SHAPE, FAKE_STEM, SPOT, SPOT_SLICE
from lesion_preservation.data import Box, load_slices, read_boxes


def test_read_boxes_flips_as_fastmri(fake_scan):
    _, csv_path = fake_scan
    row = pd.read_csv(csv_path).iloc[2]
    # get_annotation does not use self
    expected = AnnotatedSliceDataset.get_annotation(None, row, FAKE_CROP[0])
    (box,) = read_boxes(csv_path, FAKE_STEM, FAKE_CROP[0])[SPOT_SLICE]
    assert (box.row, box.col, box.height, box.width) == (
        expected["y"],
        expected["x"],
        expected["height"],
        expected["width"],
    )


def test_read_boxes_skips_study_level_and_other_files(fake_scan):
    _, csv_path = fake_scan
    boxes = read_boxes(csv_path, FAKE_STEM, FAKE_CROP[0])
    assert boxes == {
        SPOT_SLICE: [Box(2, *SPOT, "Nonspecific white matter lesion")],
    }


def test_load_slices_shapes_and_boxes(fake_scan):
    slices = list(load_slices(*fake_scan))
    assert [s.index for s in slices] == list(range(FAKE_SHAPE[0]))
    for s in slices:
        assert s.file == FAKE_STEM
        assert s.kspace.shape == FAKE_SHAPE[1:]
        assert s.reference.shape == FAKE_CROP
    assert [len(s.boxes) for s in slices] == [0, 1, 0]


def test_reference_matches_stored_reconstruction(fake_scan):
    h5_path, csv_path = fake_scan
    with h5py.File(h5_path, "r") as hf:
        stored = hf["reconstruction_rss"][()]
    for s in load_slices(h5_path, csv_path):
        np.testing.assert_allclose(s.reference, stored[s.index], rtol=1e-5)


def test_box_lands_on_spot(fake_scan):
    (s,) = load_slices(*fake_scan, slices=[SPOT_SLICE])
    (box,) = s.boxes
    inside = np.zeros(s.reference.shape, dtype=bool)
    inside[box.row : box.row + box.height, box.col : box.col + box.width] = True
    # spot is +1 on tissue between 1.0 and 1.3, with dark background around the head
    assert s.reference[inside].min() > s.reference[~inside].max() + 0.5


def test_slices_argument(fake_scan):
    assert [s.index for s in load_slices(*fake_scan, slices=[1])] == [1]
