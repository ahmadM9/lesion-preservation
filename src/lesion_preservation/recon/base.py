from typing import Protocol

import numpy as np


class Reconstructor(Protocol):
    # numpy in and out; a torch plug-in converts inside itself
    def __call__(
        self, masked_kspace: np.ndarray, mask: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray: ...
