import numpy as np

from lesion_preservation.transforms import center_crop, ifft2c, rss


class ZeroFilled:
    # same steps as the reference in data.py, so a 1x mask gives the reference exactly
    def __call__(
        self, masked_kspace: np.ndarray, mask: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        return center_crop(rss(ifft2c(masked_kspace), coil_axis=0), shape)
