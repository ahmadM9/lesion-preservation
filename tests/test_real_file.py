import csv
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import DATA_ENV
from lesion_preservation.config import load_config
from lesion_preservation.data import RECONSTRUCTION_KEY, load_slices
from lesion_preservation.lesions import boxes_from_labels, lesion_table

pytestmark = pytest.mark.requires_data

SMOKE = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def data_dir():
    return Path(os.environ[DATA_ENV])


def labelled_files():
    # the folder holds brain.csv beside the h5 files; checks every labelled file found there
    with open(data_dir() / "brain.csv", newline="") as f:
        stems = {line["file"] for line in csv.DictReader(f) if line["study_level"] == "No"}
    return sorted(p for p in data_dir().glob("*.h5") if p.stem in stems)


def test_some_labelled_file_present():
    assert labelled_files()


def test_reference_matches_stored_reconstruction():
    for h5_path in labelled_files():
        with h5py.File(h5_path, "r") as hf:
            stored = hf[RECONSTRUCTION_KEY][()]
        for s in load_slices(h5_path, data_dir() / "brain.csv"):
            error = np.abs(s.reference - stored[s.index]).max() / stored[s.index].max()
            # float32 rounding: 2.1e-07 at most on file_brain_AXFLAIR_200_6002469
            assert error < 1e-6, f"{h5_path.stem} slice {s.index}: {error:.2e}"


def test_boxes_inside_image():
    for h5_path in labelled_files():
        for s in load_slices(h5_path, data_dir() / "brain.csv"):
            rows, cols = s.reference.shape
            for box in s.boxes:
                assert 0 <= box.row and box.row + box.height <= rows, (h5_path.stem, box)
                assert 0 <= box.col and box.col + box.width <= cols, (h5_path.stem, box)


@pytest.mark.parametrize("connectivity, total, single", [(26, 851, 616), (6, 870, 625)])
def test_lesion_count_on_brain_csv(connectivity, total, single):
    # fastmri+ brain.csv: 1,699 flair white matter boxes give these counts
    cfg = load_config(SMOKE).lesions
    boxes = boxes_from_labels(data_dir() / "brain.csv", cfg.label, cfg.contrast)
    lesions = lesion_table(boxes, cfg, connectivity)
    assert len(lesions) == total
    assert sum(lesion.num_boxes == 1 for lesion in lesions) == single
