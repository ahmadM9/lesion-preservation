from lesion_preservation.recon.base import Reconstructor
from lesion_preservation.recon.zero_filled import ZeroFilled

RECONSTRUCTORS: dict[str, type] = {"zero_filled": ZeroFilled}


def get_reconstructor(name: str) -> Reconstructor:
    if name not in RECONSTRUCTORS:
        raise KeyError(f"unknown reconstruction {name!r}; known: {sorted(RECONSTRUCTORS)}")
    return RECONSTRUCTORS[name]()
