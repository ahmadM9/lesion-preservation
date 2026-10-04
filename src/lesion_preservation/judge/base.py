from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lesion_preservation.data import Box


@dataclass(frozen=True)
class Verdict:
    box_id: int
    found: bool
    score: float  # kept so the threshold can move without rerunning


class Judge(Protocol):
    # one verdict per box, in box order
    def __call__(self, image: np.ndarray, boxes: Sequence[Box]) -> list[Verdict]: ...
