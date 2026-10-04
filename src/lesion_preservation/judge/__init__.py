from lesion_preservation.judge.base import Judge, Verdict
from lesion_preservation.judge.placeholder import BoxRingContrast

JUDGES: dict[str, type] = {"placeholder": BoxRingContrast}

__all__ = ["Judge", "Verdict", "get_judge"]


def get_judge(name: str, **params) -> Judge:
    if name not in JUDGES:
        raise KeyError(f"unknown judge {name!r}; known: {sorted(JUDGES)}")
    return JUDGES[name](**params)
