import os
from pathlib import Path

import pytest

DATA_ENV = "LESION_PRESERVATION_DATA"


def pytest_collection_modifyitems(config, items):
    data_dir = os.environ.get(DATA_ENV)
    if data_dir and Path(data_dir).is_dir():
        return
    skip = pytest.mark.skip(reason=f"set {DATA_ENV} to a folder with the data files")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)
