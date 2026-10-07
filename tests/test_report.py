import csv
import math
import shutil
from pathlib import Path

import pytest

from lesion_preservation.report import report
from lesion_preservation.run import COLUMNS, run

SMOKE = Path(__file__).parents[1] / "configs" / "smoke.yaml"

# file, slice, box id, row, col, height, width
BOXES = [
    # one lesion on two slices
    ("a", 0, 0, 10, 10, 4, 4),
    ("a", 1, 1, 10, 10, 4, 4),
    # corner contact across slices: one lesion at 26, two at 6
    ("a", 3, 2, 30, 30, 4, 4),
    ("a", 4, 3, 34, 34, 4, 4),
    # sizes at the bin edges of smoke.yaml (<=7, 8-16, >16); two boxes share slice 0
    ("b", 0, 4, 0, 0, 7, 1),
    ("b", 0, 5, 0, 50, 8, 1),
    ("b", 4, 6, 0, 0, 16, 1),
    ("b", 6, 7, 0, 0, 60, 1),
    ("b", 8, 8, 0, 0, 17, 1),
]


def write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copyfile(SMOKE, run_dir / "config.yaml")
    rows = []
    for speedup in (1.0, 2.0):
        for file, index, box_id, row, col, height, width in BOXES:
            # at 2x only box 0 is found; slice 0 of b scores lower than the others
            psnr = math.inf if speedup == 1 else (10.0 if (file, index) == ("b", 0) else 20.0)
            values = {
                "file": file, "slice": index, "box_id": box_id, "box_row": row,
                "box_col": col, "box_height": height, "box_width": width, "label": "x",
                "kind": "lesion",
                "method": "m", "speedup": speedup, "effective_speedup": speedup,
                "found": speedup == 1 or box_id == 0, "slice_psnr": psnr,
                "slice_ssim": 1.0 if speedup == 1 else 0.5, "slice_nmse": 0.0,
                "box_psnr": psnr, "box_ssim": 1.0,
            }  # fmt: skip
            rows.append(dict.fromkeys(COLUMNS, "") | values)
            # a normal box beside each lesion box, same id and size; found only at 1x
            rows.append(
                dict.fromkeys(COLUMNS, "")
                | values
                | {"box_col": col + 100, "label": "", "kind": "normal", "found": speedup == 1}
            )
    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pick(rows, speedup, unit, size_bin, connectivity=""):
    (row,) = [
        r for r in rows
        if float(r["speedup"]) == speedup and r["unit"] == unit and r["size_bin"] == size_bin
        and r["connectivity"] == str(connectivity)
    ]  # fmt: skip
    return row


@pytest.fixture
def reported(tmp_path):
    run_dir = write_run(tmp_path)
    out = report(run_dir)
    return run_dir, out


def test_files(reported):
    _, out = reported
    assert sorted(p.name for p in out.iterdir()) == [
        "found_rates.csv", "report_m.png", "scores.csv",
    ]  # fmt: skip


def test_lesion_found_on_one_slice(reported):
    _, out = reported
    rows = read(out / "found_rates.csv")
    # lesions <=7 at 26: the two-slice lesion (found on slice 0), the corner pair, box 4
    lesion = pick(rows, 2, "lesion", "<=7", 26)
    assert (lesion["n"], lesion["n_found"]) == ("3", "1")
    # the five boxes <=7: only box 0 found
    box = pick(rows, 2, "box", "<=7")
    assert (box["n"], box["n_found"], float(box["found_rate"])) == ("5", "1", 0.2)


def test_corner_contact_joins_at_26_not_6(reported):
    _, out = reported
    rows = read(out / "found_rates.csv")
    assert pick(rows, 2, "lesion", "<=7", 26)["n"] == "3"
    assert pick(rows, 2, "lesion", "<=7", 6)["n"] == "4"


def test_bin_edges_and_large(reported):
    _, out = reported
    rows = read(out / "found_rates.csv")
    assert pick(rows, 1, "lesion", "8-16", 26)["n"] == "2"  # sizes 8 and 16
    large = pick(rows, 1, "lesion", ">16", 26)  # sizes 17 and 60
    assert (large["n"], large["n_large"], float(large["found_rate"])) == ("2", "1", 1.0)


def test_each_slice_counts_once(reported):
    _, out = reported
    rows = {float(r["speedup"]): r for r in read(out / "scores.csv")}
    # 8 slices for 9 boxes: b slice 0 holds two boxes but counts once
    assert rows[2]["n_slices"] == "8"
    assert float(rows[2]["slice_psnr"]) == pytest.approx((7 * 20 + 10) / 8)


def test_exact_copy_kept_as_inf(reported):
    _, out = reported
    rows = {float(r["speedup"]): r for r in read(out / "scores.csv")}
    assert math.isinf(float(rows[1]["slice_psnr"]))


def test_no_overwrite(reported):
    run_dir, _ = reported
    with pytest.raises(FileExistsError):
        report(run_dir)


def test_smoke_run(fake_scan, tmp_path):
    h5_path, csv_path = fake_scan
    run_dir = run(SMOKE, h5_path.parent, csv_path, runs_dir=tmp_path / "runs")
    out = report(run_dir)
    rows = read(out / "found_rates.csv")
    # the planted 4 x 3 spot is one lesion, found at 1x
    lesion = pick(rows, 1, "lesion", "<=7", 26)
    assert (lesion["n"], lesion["found_rate"]) == ("1", "1.0")
    assert (out / "report_zero_filled.png").exists()


def test_normal_boxes_kept_apart(reported):
    _, out = reported
    rows = read(out / "found_rates.csv")
    # same size bins as the lesion boxes, none found at 2x; the lesion tables above are unchanged
    normal = pick(rows, 2, "normal_box", "<=7")
    assert (normal["n"], normal["n_found"]) == ("5", "0")
    assert pick(rows, 1, "normal_box", ">16")["found_rate"] == "1.0"
