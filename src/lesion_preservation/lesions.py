"""Count lesions in a fastMRI+ label file.

python -m lesion_preservation.lesions configs/smoke.yaml --labels-csv <path to brain.csv>
"""

import argparse
import csv
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from lesion_preservation.config import LesionsConfig, load_config
from lesion_preservation.data import Box


@dataclass(frozen=True)
class Lesion:
    lesion_id: int
    file: str
    slices: tuple[int, ...]
    num_boxes: int
    size: int  # longest box side among the lesion's boxes, pixels
    size_bin: str
    large: bool


# voxel neighbours that count as touching, as scipy's structure rank: face, plus edge, plus corner
CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


def join(boxes: Iterable[tuple[str, int, Box]], connectivity: int) -> dict[int, int]:
    # lesions are the 3d connected components of the boxes filled in as solid blocks, as in the
    # isles'22 and brats 2023 lesion-wise metrics
    structure = ndimage.generate_binary_structure(3, CONNECTIVITY_RANK[connectivity])
    by_file: dict[str, list[tuple[int, Box]]] = {}
    for file, index, box in boxes:
        by_file.setdefault(file, []).append((index, box))
    lesion_of = {}
    for in_file in by_file.values():
        shape = (
            max(index for index, _ in in_file) + 1,
            max(box.row + box.height for _, box in in_file),
            max(box.col + box.width for _, box in in_file),
        )
        mask = np.zeros(shape, dtype=bool)
        for index, box in in_file:
            mask[index, box.row : box.row + box.height, box.col : box.col + box.width] = True
        components, _ = ndimage.label(mask, structure)
        # a filled box is connected, so its corner pixel names its whole component
        component_of = {box.box_id: components[index, box.row, box.col] for index, box in in_file}
        for component in set(component_of.values()):
            members = [box_id for box_id, c in component_of.items() if c == component]
            # the smallest box id names the lesion: stable and traceable to a csv row
            lesion_of |= dict.fromkeys(members, min(members))
    return lesion_of


def size_bin(size: int, upper_bounds: Sequence[int]) -> str:
    # integer upper bounds, inclusive: [7, 16] gives <=7, 8-16, >16
    i = bisect_left(upper_bounds, size)
    if i == 0:
        return f"<={upper_bounds[0]}"
    if i == len(upper_bounds):
        return f">{upper_bounds[-1]}"
    return f"{upper_bounds[i - 1] + 1}-{upper_bounds[i]}"


def lesion_table(
    boxes: Iterable[tuple[str, int, Box]], cfg: LesionsConfig, connectivity: int
) -> list[Lesion]:
    items = list(boxes)
    lesion_of = join(items, connectivity)
    groups: dict[int, list[tuple[str, int, Box]]] = {}
    for item in items:
        groups.setdefault(lesion_of[item[2].box_id], []).append(item)
    lesions = []
    for lesion_id, members in sorted(groups.items()):
        size = max(max(box.height, box.width) for _, _, box in members)
        lesions.append(
            Lesion(
                lesion_id=lesion_id,
                file=members[0][0],
                slices=tuple(sorted({s for _, s, _ in members})),
                num_boxes=len(members),
                size=size,
                size_bin=size_bin(size, cfg.size_upper_bounds),
                large=size > cfg.large_box_px,
            )
        )
    return lesions


def boxes_from_labels(
    csv_path: str | Path, label: str, contrast: str
) -> list[tuple[str, int, Box]]:
    boxes = []
    with open(csv_path, newline="") as f:
        # box id is the data row, counted from 0, as in data.read_boxes
        for box_id, line in enumerate(csv.DictReader(f)):
            if line["study_level"] == "Yes" or line["label"] != label:
                continue
            if f"_{contrast}_" not in line["file"]:
                continue
            # y is not flipped: the image height is in the scan file, and flipping every box of
            # one file by the same height does not change which boxes touch
            box = Box(
                box_id, int(line["y"]), int(line["x"]), int(line["height"]), int(line["width"]),
                line["label"],
            )  # fmt: skip
            boxes.append((line["file"], int(line["slice"]), box))
    return boxes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count lesions in a fastMRI+ label file")
    parser.add_argument("config", type=Path)
    parser.add_argument("--labels-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config).lesions
    boxes = boxes_from_labels(args.labels_csv, cfg.label, cfg.contrast)
    for connectivity in (cfg.connectivity, cfg.connectivity_variant):
        lesions = lesion_table(boxes, cfg, connectivity)
        print(f"connectivity {connectivity}")
        print(f"  lesions {len(lesions)}")
        print(f"  single box {sum(lesion.num_boxes == 1 for lesion in lesions)}")
        print(f"  large {sum(lesion.large for lesion in lesions)}")
        bounds = cfg.size_upper_bounds
        # each bound falls in its own bin, so this lists the bins smallest first
        for size in [*bounds, bounds[-1] + 1]:
            name = size_bin(size, bounds)
            print(f"  bin {name} {sum(lesion.size_bin == name for lesion in lesions)}")


if __name__ == "__main__":
    main()
