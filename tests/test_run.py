import csv
import math
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from conftest import FAKE_STEM, SPOT_SLICE
from lesion_preservation.run import COLUMNS, run

SMOKE = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def read_rows(run_dir):
    with open(run_dir / "results.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def smoke_run(fake_scan, tmp_path):
    h5_path, csv_path = fake_scan
    run_dir = run(SMOKE, h5_path.parent, csv_path, runs_dir=tmp_path / "runs")
    return run_dir, h5_path, csv_path


def test_fake_scan_gives_one_row_per_speedup(smoke_run):
    run_dir, _, _ = smoke_run
    rows = read_rows(run_dir)
    assert len(rows) == 4
    assert list(rows[0]) == COLUMNS
    assert {r["file"] for r in rows} == {FAKE_STEM}
    assert {int(r["slice"]) for r in rows} == {SPOT_SLICE}
    assert sorted(float(r["speedup"]) for r in rows) == [1, 2, 4, 8]


def test_full_scan_row(smoke_run):
    run_dir, _, _ = smoke_run
    (row,) = [r for r in read_rows(run_dir) if float(r["speedup"]) == 1]
    assert row["found"] == "True"
    assert math.isinf(float(row["slice_psnr"]))
    assert float(row["effective_speedup"]) == 1


def test_run_folder_contents(smoke_run):
    run_dir, _, _ = smoke_run
    for name in ["config.yaml", "code_version.txt", "run_record.yaml", "masks.npz"]:
        assert (run_dir / name).exists()
    examples = sorted(p.name for p in (run_dir / "examples").iterdir())
    assert examples == [f"{FAKE_STEM}_s{SPOT_SLICE}_{s}x_zero_filled.png" for s in (1, 2, 4, 8)]
    (session,) = yaml.safe_load((run_dir / "run_record.yaml").read_text())
    assert session["runtime"].count(":") == 2


def test_same_run_twice_fails(smoke_run, tmp_path):
    _, h5_path, csv_path = smoke_run
    with pytest.raises(FileExistsError):
        run(SMOKE, h5_path.parent, csv_path, runs_dir=tmp_path / "runs")


def test_resume_writes_only_missing_items(smoke_run):
    run_dir, h5_path, csv_path = smoke_run
    lines = (run_dir / "results.csv").read_text().splitlines(keepends=True)
    (run_dir / "results.csv").write_text("".join(lines[:-1]))
    run(SMOKE, h5_path.parent, csv_path, resume=run_dir)
    rows = read_rows(run_dir)
    assert len(rows) == 4
    assert len({r["speedup"] for r in rows}) == 4
    assert len(yaml.safe_load((run_dir / "run_record.yaml").read_text())) == 2


def test_resume_with_changed_config_fails(smoke_run, tmp_path):
    run_dir, h5_path, csv_path = smoke_run
    raw = yaml.safe_load(SMOKE.read_text())
    raw["mask"]["seed"] = 1
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="differs"):
        run(changed, h5_path.parent, csv_path, resume=run_dir)


def git(folder, *args):
    # identity given per call, so no git config file is changed
    subprocess.run(
        ["git", "-C", str(folder), "-c", "user.name=test", "-c", "user.email=test@test", *args],
        capture_output=True,
        check=True,
    )


def test_code_version_names_both_repos(smoke_run):
    run_dir, _, _ = smoke_run
    lines = (run_dir / "code_version.txt").read_text().splitlines()
    assert [line.split(": ")[0] for line in lines] == ["pipeline", "config"]


def test_config_outside_git(fake_scan, tmp_path):
    h5_path, csv_path = fake_scan
    config = tmp_path / "plain" / "smoke.yaml"
    config.parent.mkdir()
    shutil.copyfile(SMOKE, config)
    run_dir = run(config, h5_path.parent, csv_path, runs_dir=tmp_path / "runs")
    lines = (run_dir / "code_version.txt").read_text().splitlines()
    assert lines[1] == "config: not a git checkout"


def test_resume_after_config_repo_change_fails(fake_scan, tmp_path):
    h5_path, csv_path = fake_scan
    repo = tmp_path / "study"
    repo.mkdir()
    git(repo, "init", "-q")
    shutil.copyfile(SMOKE, repo / "smoke.yaml")
    git(repo, "add", "smoke.yaml")
    git(repo, "commit", "-q", "-m", "add config")
    run_dir = run(repo / "smoke.yaml", h5_path.parent, csv_path, runs_dir=tmp_path / "runs")
    git(repo, "commit", "-q", "--allow-empty", "-m", "move on")
    with pytest.raises(ValueError, match="code changed"):
        run(repo / "smoke.yaml", h5_path.parent, csv_path, resume=run_dir)


def test_other_labels_not_scored(fake_scan, tmp_path):
    h5_path, csv_path = fake_scan
    lines = csv_path.read_text().splitlines()
    other = lines[-1].replace("Nonspecific white matter lesion", "Posttreatment change")
    csv_path.write_text("\n".join([*lines, other]) + "\n")
    rows = read_rows(run(SMOKE, h5_path.parent, csv_path, runs_dir=tmp_path / "runs"))
    assert len(rows) == 4
    assert {r["label"] for r in rows} == {"Nonspecific white matter lesion"}
