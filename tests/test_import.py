import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["lesion_preservation", "numpy", "skimage", "h5py", "yaml", "torch"]
)
def test_import(module):
    importlib.import_module(module)
