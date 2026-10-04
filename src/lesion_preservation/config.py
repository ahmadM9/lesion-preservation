from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lesion_preservation.judge import get_judge
from lesion_preservation.masks import get_mask
from lesion_preservation.recon import get_reconstructor


@dataclass(frozen=True)
class MaskConfig:
    name: str
    centre_fraction: float
    seed: int


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ExamplesConfig:
    save: bool
    scale: int


@dataclass(frozen=True)
class LesionsConfig:
    label: str
    contrast: str
    connectivity: int
    connectivity_variant: int
    size_upper_bounds: tuple[int, ...]
    large_box_px: int


@dataclass(frozen=True)
class Config:
    name: str
    files: tuple[str, ...]
    mask: MaskConfig
    speedups: tuple[float, ...]
    methods: tuple[str, ...]
    judge: JudgeConfig
    ssim_window: int
    examples: ExamplesConfig
    lesions: LesionsConfig


def _check_keys(section: dict, keys: set[str], where: str) -> None:
    # no defaults: every key must be present, and a misspelled key must not be ignored
    if not isinstance(section, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(section).__name__}")
    missing = sorted(keys - section.keys())
    unknown = sorted(section.keys() - keys)
    if missing:
        raise KeyError(f"{where}: missing key(s) {missing}")
    if unknown:
        raise KeyError(f"{where}: unknown key(s) {unknown}")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    _check_keys(raw, set(Config.__dataclass_fields__), str(path))
    _check_keys(raw["mask"], set(MaskConfig.__dataclass_fields__), f"{path}: mask")
    _check_keys(raw["judge"], set(JudgeConfig.__dataclass_fields__), f"{path}: judge")
    _check_keys(raw["examples"], set(ExamplesConfig.__dataclass_fields__), f"{path}: examples")
    _check_keys(raw["lesions"], set(LesionsConfig.__dataclass_fields__), f"{path}: lesions")

    cfg = Config(
        name=raw["name"],
        files=tuple(raw["files"]),
        mask=MaskConfig(**raw["mask"]),
        speedups=tuple(float(s) for s in raw["speedups"]),
        methods=tuple(raw["methods"]),
        judge=JudgeConfig(**raw["judge"]),
        ssim_window=raw["ssim_window"],
        examples=ExamplesConfig(**raw["examples"]),
        lesions=LesionsConfig(
            **raw["lesions"] | {"size_upper_bounds": tuple(raw["lesions"]["size_upper_bounds"])}
        ),
    )
    for key in ("connectivity", "connectivity_variant"):
        if getattr(cfg.lesions, key) not in (6, 18, 26):
            raise ValueError(f"{path}: lesions {key} must be 6, 18 or 26")
    # resolve every name now, so a wrong one fails before the run starts
    get_mask(cfg.mask.name)
    for method in cfg.methods:
        get_reconstructor(method)
    try:
        get_judge(cfg.judge.name, **cfg.judge.params)
    except TypeError as err:
        raise KeyError(f"{path}: judge params do not fit {cfg.judge.name!r}: {err}") from err
    return cfg
