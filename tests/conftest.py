import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from lesion_preservation.transforms import center_crop, fft2c, ifft2c, rss

DATA_ENV = "LESION_PRESERVATION_DATA"


def pytest_collection_modifyitems(config, items):
    data_dir = os.environ.get(DATA_ENV)
    if data_dir and Path(data_dir).is_dir():
        return
    skip = pytest.mark.skip(reason=f"set {DATA_ENV} to a folder with the data files")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


FAKE_STEM = "file_brain_AXFLAIR_000_0000000"
FAKE_SHAPE = (3, 4, 32, 24)  # slices, coils, rows, cols; rows != cols catches swapped axes
FAKE_CROP = (16, 16)
SPOT_SLICE = 1
# top row, left col, height, width inside the cropped image; not square on purpose
SPOT = (3, 9, 4, 3)


def fake_coil_maps(num_coils, rows, cols):
    # smooth maps scaled so their summed squared magnitude is 1: rss then equals |image|
    r, c = np.mgrid[:rows, :cols]
    maps = []
    for k in range(num_coils):
        angle = 2 * np.pi * k / num_coils
        centre_r = rows / 2 + rows / 3 * np.cos(angle)
        centre_c = cols / 2 + cols / 3 * np.sin(angle)
        weight = np.exp(-((r - centre_r) ** 2 + (c - centre_c) ** 2) / (2 * (rows / 3) ** 2))
        maps.append(weight * np.exp(1j * angle))
    maps = np.array(maps)
    return maps / np.sqrt(np.sum(np.abs(maps) ** 2, axis=0))


@pytest.fixture
def fake_scan(tmp_path):
    num_slices, num_coils, rows, cols = FAKE_SHAPE
    r, c = np.mgrid[:rows, :cols]
    images = np.repeat((1.0 + 0.2 * r / rows + 0.1 * c / cols)[None], num_slices, axis=0)
    top = SPOT[0] + (rows - FAKE_CROP[0]) // 2
    left = SPOT[1] + (cols - FAKE_CROP[1]) // 2
    images[SPOT_SLICE, top : top + SPOT[2], left : left + SPOT[3]] += 4.0

    kspace = fft2c(images[:, None] * fake_coil_maps(num_coils, rows, cols)).astype(np.complex64)
    stored = center_crop(rss(ifft2c(kspace), coil_axis=1), FAKE_CROP).astype(np.float32)

    h5_path = tmp_path / f"{FAKE_STEM}.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("reconstruction_rss", data=stored)

    # fastmri+ layout, y counted from the bottom of the cropped image
    y = FAKE_CROP[0] - SPOT[0] - SPOT[2]
    csv_path = tmp_path / "brain.csv"
    csv_path.write_text(
        "file,slice,study_level,x,y,width,height,label\n"
        "file_brain_AXFLAIR_000_9999999,1,No,1,1,2,2,Nonspecific white matter lesion\n"
        f"{FAKE_STEM},0,Yes,,,,,Normal for age\n"
        f"{FAKE_STEM},{SPOT_SLICE},No,{SPOT[1]},{y},{SPOT[3]},{SPOT[2]},"
        "Nonspecific white matter lesion\n"
    )
    return h5_path, csv_path
