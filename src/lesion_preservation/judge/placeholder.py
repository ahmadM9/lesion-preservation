from collections.abc import Sequence

import numpy as np

from lesion_preservation.data import Box
from lesion_preservation.judge.base import Verdict


class BoxRingContrast:
    # for building only, never reported: found if the box is clearly brighter than a ring
    # of tissue around it
    def __init__(self, ring_width: int, threshold: float):
        self.ring_width = ring_width
        self.threshold = threshold

    def score(self, image: np.ndarray, box: Box) -> float:
        rows = slice(box.row, box.row + box.height)
        cols = slice(box.col, box.col + box.width)
        # the box grown on every side, clipped at the image edge
        w = self.ring_width
        outer_rows = slice(max(box.row - w, 0), box.row + box.height + w)
        outer_cols = slice(max(box.col - w, 0), box.col + box.width + w)
        in_ring = np.zeros(image.shape, dtype=bool)
        in_ring[outer_rows, outer_cols] = True
        in_ring[rows, cols] = False
        # a ring of zeros, as outside the head, gives an infinite score
        with np.errstate(divide="ignore"):
            return float(image[rows, cols].mean() / image[in_ring].mean())

    def __call__(self, image: np.ndarray, boxes: Sequence[Box]) -> list[Verdict]:
        verdicts = []
        for box in boxes:
            score = self.score(image, box)
            verdicts.append(Verdict(box.box_id, score >= self.threshold, score))
        return verdicts
