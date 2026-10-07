from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from lesion_preservation.data import Box


def brain_mask(reference: np.ndarray, margin: int) -> np.ndarray:
    # head against background by otsu; erosion takes off the scalp and the brain's edge
    head = ndimage.binary_fill_holes(reference > threshold_otsu(reference))
    if margin > 0:
        head = ndimage.binary_erosion(head, iterations=margin)
    pieces, count = ndimage.label(head)
    if count == 0:
        return head
    sizes = np.bincount(pieces.ravel())[1:]
    return pieces == 1 + int(np.argmax(sizes))


def window_counts(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    # true pixels in the height x width window at each top-left corner
    total = np.pad(mask, ((1, 0), (1, 0))).astype(np.int64).cumsum(0).cumsum(1)
    return (
        total[height:, width:]
        - total[:-height, width:]
        - total[height:, :-width]
        + total[:-height, :-width]
    )


def place_normal_boxes(
    reference: np.ndarray,
    labelled: Sequence[Box],
    lesions: Sequence[Box],
    seed: int,
    margin: int,
    clearance: int,
    max_dark_fraction: float,
    neighbours: Sequence[np.ndarray],
) -> list[Box]:
    # one per lesion box, same size, keeping its id; a lesion box with no free place gets none
    allowed = brain_mask(reference, margin)
    # below otsu is fluid or background: the mask holds ventricles and, on top slices, the gap
    # under the skull, so a box mostly dark would compare a lesion with fluid, not tissue
    dark = reference <= threshold_otsu(reference)
    # a thick slice grazing a ventricle averages its fluid with tissue and shows grey; the
    # fluid is still dark on the next slice, which shares the pixel grid
    for image in neighbours:
        dark |= image <= threshold_otsu(image)
    # every label is kept clear, not only the lesions scored
    for box in labelled:
        allowed[box.row : box.row + box.height, box.col : box.col + box.width] = False
    normals = []
    for lesion in lesions:
        # the box grown by the clearance must fit, so its surroundings are brain too
        height, width = lesion.height + 2 * clearance, lesion.width + 2 * clearance
        free = window_counts(~allowed, height, width) == 0
        tissue = window_counts(dark, lesion.height, lesion.width) <= max_dark_fraction * (
            lesion.height * lesion.width
        )
        # a footprint corner sits clearance pixels above and left of its box's corner
        free &= tissue[clearance : clearance + free.shape[0], clearance : clearance + free.shape[1]]
        corners = np.argwhere(free)
        if len(corners) == 0:
            continue
        # seeded by box id, unique in the label csv, so a box's place does not depend on the run
        top, left = corners[np.random.default_rng([seed, lesion.box_id]).integers(len(corners))]
        row, col = int(top) + clearance, int(left) + clearance
        normals.append(Box(lesion.box_id, row, col, lesion.height, lesion.width, ""))
        allowed[row : row + lesion.height, col : col + lesion.width] = False
    return normals
