from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lesion_preservation.data import Box

# outline colour only; it changes no result
BOX_COLOUR = (255, 0, 0)
NORMAL_COLOUR = (0, 255, 0)


def to_grey(image: np.ndarray, data_range: float) -> np.ndarray:
    # same scale for reference and reconstruction, so a fainter lesion looks fainter
    return (np.clip(image / data_range, 0, 1) * 255).astype(np.uint8)


def save_example(
    reference: np.ndarray,
    image: np.ndarray,
    boxes: Sequence[Box],
    data_range: float,
    scale: int,
    path: str | Path,
    normal_boxes: Sequence[Box] = (),
) -> None:
    # reference on the left, reconstruction on the right, each with its box outlines
    grey = np.concatenate([to_grey(reference, data_range), to_grey(image, data_range)], axis=1)
    rows, cols = grey.shape
    # nearest neighbour, so small boxes are enlarged without blurring
    picture = Image.fromarray(grey).resize((cols * scale, rows * scale), Image.NEAREST)
    picture = picture.convert("RGB")
    draw = ImageDraw.Draw(picture)
    width = reference.shape[1]
    for offset in (0, width):
        for colour, group in ((BOX_COLOUR, boxes), (NORMAL_COLOUR, normal_boxes)):
            for box in group:
                left = (box.col + offset) * scale
                top = box.row * scale
                # one pixel outside the box, so the lesion pixels stay visible
                draw.rectangle(
                    (left - 1, top - 1, left + box.width * scale, top + box.height * scale),
                    outline=colour,
                )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    picture.save(path)
