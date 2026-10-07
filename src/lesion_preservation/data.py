import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from lesion_preservation.transforms import center_crop, ifft2c, rss

# dataset names inside a fastmri brain file, as read by fastmri 0.3.0 data/mri_data.py
KSPACE_KEY = "kspace"
RECONSTRUCTION_KEY = "reconstruction_rss"


@dataclass(frozen=True)
class Box:
    # box_id is the data row of the box in the label csv, counted from 0
    box_id: int
    row: int
    col: int
    height: int
    width: int
    label: str


@dataclass
class Slice:
    file: str
    index: int
    kspace: np.ndarray  # (coils, rows, cols), fully sampled
    reference: np.ndarray  # rss of the full k-space, cropped to the stored reconstruction size
    boxes: list[Box] = field(default_factory=list)


def read_boxes(csv_path: str | Path, file_stem: str, num_rows: int) -> dict[int, list[Box]]:
    boxes: dict[int, list[Box]] = {}
    with open(csv_path, newline="") as f:
        for box_id, line in enumerate(csv.DictReader(f)):
            # study-level labels belong to the whole scan and have no box
            if line["file"] != file_stem or line["study_level"] == "Yes":
                continue
            height = int(line["height"])
            # y in the csv counts from the bottom; same flip as fastmri 0.3.0 get_annotation
            row = num_rows - int(line["y"]) - height
            box = Box(box_id, row, int(line["x"]), height, int(line["width"]), line["label"])
            boxes.setdefault(int(line["slice"]), []).append(box)
    return boxes


def with_neighbours(slices: Iterable[Slice], reach: int) -> Iterator[tuple[Slice, list]]:
    # each slice with the references of up to reach slices on either side; reads ahead by reach
    ahead: list[Slice] = []
    references: dict[int, np.ndarray] = {}

    def neighbours(s: Slice) -> list[np.ndarray]:
        near = range(s.index - reach, s.index + reach + 1)
        return [references[i] for i in near if i != s.index and i in references]

    for s in slices:
        references[s.index] = s.reference
        ahead.append(s)
        if len(ahead) > reach:
            current = ahead.pop(0)
            yield current, neighbours(current)
    for current in ahead:
        yield current, neighbours(current)


def load_slices(
    h5_path: str | Path, csv_path: str | Path, slices: Sequence[int] | None = None
) -> Iterator[Slice]:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as hf:
        kspace_all = hf[KSPACE_KEY]
        # crop size from the stored reconstruction, so the header xml need not be parsed
        crop = hf[RECONSTRUCTION_KEY].shape[-2:]
        boxes = read_boxes(csv_path, h5_path.stem, num_rows=crop[0])
        indices = range(kspace_all.shape[0]) if slices is None else slices
        for index in indices:
            # one slice at a time: real files are too large to hold whole
            kspace = kspace_all[index]
            reference = center_crop(rss(ifft2c(kspace), coil_axis=0), crop)
            yield Slice(h5_path.stem, index, kspace, reference, boxes.get(index, []))
