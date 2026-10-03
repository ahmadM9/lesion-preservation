import numpy as np
from skimage.metrics import structural_similarity

from lesion_preservation.data import Box

# data_range is the slice reference maximum for slice and box scores alike, so a box is
# scored on the same scale as its slice; fastmri uses the volume maximum


def nmse(ref: np.ndarray, img: np.ndarray) -> float:
    return float(np.linalg.norm(ref - img) ** 2 / np.linalg.norm(ref) ** 2)


def psnr(ref: np.ndarray, img: np.ndarray, data_range: float) -> float:
    mse = np.mean((ref - img) ** 2)
    with np.errstate(divide="ignore"):
        return float(10 * np.log10(data_range**2 / mse))


def ssim(
    ref: np.ndarray, img: np.ndarray, data_range: float, win_size: int
) -> tuple[float, np.ndarray]:
    mean, ssim_map = structural_similarity(
        ref, img, data_range=data_range, win_size=win_size, full=True
    )
    return float(mean), ssim_map


def box_scores(
    ref: np.ndarray, img: np.ndarray, ssim_map: np.ndarray, box: Box, data_range: float
) -> dict[str, float]:
    # ssim is the slice map averaged over the box: works for boxes smaller than the window,
    # but each value also sees tissue around the box
    rows = slice(box.row, box.row + box.height)
    cols = slice(box.col, box.col + box.width)
    return {
        "psnr": psnr(ref[rows, cols], img[rows, cols], data_range),
        "ssim": float(ssim_map[rows, cols].mean()),
        "nmse": nmse(ref[rows, cols], img[rows, cols]),
    }
