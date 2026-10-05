from pathlib import Path

import pytest
import yaml

from lesion_preservation.config import load_config

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))
SMOKE = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def write_changed(tmp_path, change):
    raw = yaml.safe_load(SMOKE.read_text())
    change(raw)
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_config_loads(path):
    load_config(path)


def test_missing_key_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw.pop("ssim_window"))
    with pytest.raises(KeyError, match="ssim_window"):
        load_config(path)


def test_missing_nested_key_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw["mask"].pop("seed"))
    with pytest.raises(KeyError, match="seed"):
        load_config(path)


def test_missing_examples_key_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw["examples"].pop("scale"))
    with pytest.raises(KeyError, match="scale"):
        load_config(path)


def test_missing_lesions_key_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw["lesions"].pop("connectivity"))
    with pytest.raises(KeyError, match="connectivity"):
        load_config(path)


def test_unknown_connectivity_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw["lesions"].update(connectivity=8))
    with pytest.raises(ValueError, match="6, 18 or 26"):
        load_config(path)


def test_unknown_key_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw.update(speedup=[2]))
    with pytest.raises(KeyError, match="unknown"):
        load_config(path)


def test_unknown_method_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw.update(methods=["varnet"]))
    with pytest.raises(KeyError, match="varnet"):
        load_config(path)


def test_wrong_judge_params_fail(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw["judge"]["params"].pop("threshold"))
    with pytest.raises(KeyError, match="judge params"):
        load_config(path)


def test_file_of_other_contrast_fails(tmp_path):
    path = write_changed(tmp_path, lambda raw: raw.update(files=["file_brain_AXT2_200_6002469"]))
    with pytest.raises(ValueError, match="contrast"):
        load_config(path)
