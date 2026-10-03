import numpy as np

# all functions act on the last two axes, so a leading coil axis passes through untouched
_IMAGE_AXES = (-2, -1)


def fft2c(image: np.ndarray) -> np.ndarray:
    # centred: the k-space centre (low frequencies) sits in the middle of the array
    shifted = np.fft.ifftshift(image, axes=_IMAGE_AXES)
    kspace = np.fft.fftn(shifted, axes=_IMAGE_AXES, norm="ortho")
    return np.fft.fftshift(kspace, axes=_IMAGE_AXES)


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=_IMAGE_AXES)
    image = np.fft.ifftn(shifted, axes=_IMAGE_AXES, norm="ortho")
    return np.fft.fftshift(image, axes=_IMAGE_AXES)


def rss(coil_images: np.ndarray, coil_axis: int) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=coil_axis))


def center_crop(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = image.shape[-2:]
    if not (0 < shape[0] <= rows and 0 < shape[1] <= cols):
        raise ValueError(f"cannot crop {(rows, cols)} to {tuple(shape)}")
    top = (rows - shape[0]) // 2
    left = (cols - shape[1]) // 2
    return image[..., top : top + shape[0], left : left + shape[1]]
