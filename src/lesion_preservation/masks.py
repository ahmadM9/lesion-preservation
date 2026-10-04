import numpy as np


def random_mask(
    num_cols: int, speedup: float, center_fraction: float, seed: int | tuple[int, ...]
) -> np.ndarray:
    # python round (half to even) as in fastmri, so the centre block has the same width
    num_low = round(num_cols * center_fraction)
    mask = np.zeros(num_cols, dtype=bool)
    start = (num_cols - num_low + 1) // 2
    mask[start : start + num_low] = True

    # legacy RandomState on purpose: same masks as fastmri seed for seed, and its stream
    # is frozen across numpy versions
    rng = np.random.RandomState(seed)
    # chosen so that the expected number of kept lines is num_cols / speedup
    prob = (num_cols / speedup - num_low) / (num_cols - num_low)
    return mask | (rng.uniform(size=num_cols) < prob)


def apply_mask(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # mask runs along the last axis (columns); + 0.0 turns -0.0 into 0.0, as fastmri does
    return kspace * mask + 0.0


def effective_speedup(mask: np.ndarray) -> float:
    return mask.size / int(mask.sum())


MASKS = {"random": random_mask}


def get_mask(name: str):
    if name not in MASKS:
        raise KeyError(f"unknown mask {name!r}; known: {sorted(MASKS)}")
    return MASKS[name]
