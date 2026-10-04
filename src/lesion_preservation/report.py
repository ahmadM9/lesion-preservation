"""Report a finished run: found-rate per size group and speed-up, scores beside.

python -m lesion_preservation.report <run folder>
"""

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from lesion_preservation.config import LesionsConfig, load_config
from lesion_preservation.data import Box
from lesion_preservation.lesions import join, lesion_table, size_bin

FOUND_COLUMNS = [
    "method", "speedup", "unit", "connectivity", "size_bin", "n", "n_found", "found_rate",
    "n_large", "box_psnr", "box_ssim",
]  # fmt: skip
SCORE_COLUMNS = [
    "method", "speedup", "n_slices", "effective_speedup", "slice_psnr", "slice_ssim",
    "slice_nmse",
]  # fmt: skip


def read_results(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def boxes_of(rows: list[dict]) -> list[tuple[str, int, Box]]:
    # each box appears once per speed-up and method; keep it once
    boxes = {}
    for r in rows:
        box = Box(
            int(r["box_id"]), int(r["box_row"]), int(r["box_col"]), int(r["box_height"]),
            int(r["box_width"]), r["label"],
        )  # fmt: skip
        boxes[box.box_id] = (r["file"], int(r["slice"]), box)
    return list(boxes.values())


def bin_names(upper_bounds: Sequence[int]) -> list[str]:
    # each bound falls in its own bin, so this lists the bins smallest first
    return [size_bin(size, upper_bounds) for size in [*upper_bounds, upper_bounds[-1] + 1]]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def found_rates(rows: list[dict], cfg: LesionsConfig) -> list[dict]:
    boxes = boxes_of(rows)
    box_size = {box.box_id: max(box.height, box.width) for _, _, box in boxes}
    found = {
        (r["method"], float(r["speedup"]), int(r["box_id"])): r["found"] == "True" for r in rows
    }
    by_item: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        by_item.setdefault((r["method"], float(r["speedup"])), []).append(r)
    tables = {c: lesion_table(boxes, cfg, c) for c in (cfg.connectivity, cfg.connectivity_variant)}
    # box ids of each lesion, per connectivity
    members: dict[int, dict[int, list[int]]] = {c: {} for c in tables}
    for connectivity, groups in members.items():
        for box_id, lesion_id in join(boxes, connectivity).items():
            groups.setdefault(lesion_id, []).append(box_id)

    out = []
    for (method, speedup), item_rows in sorted(by_item.items()):
        for name in bin_names(cfg.size_upper_bounds):
            # per lesion: found if any of its boxes is found, on any slice
            for connectivity, lesions in tables.items():
                in_bin = [lesion for lesion in lesions if lesion.size_bin == name]
                n_found = sum(
                    any(found[method, speedup, b] for b in members[connectivity][lesion.lesion_id])
                    for lesion in in_bin
                )
                out.append({
                    "method": method, "speedup": speedup, "unit": "lesion",
                    "connectivity": connectivity, "size_bin": name, "n": len(in_bin),
                    "n_found": n_found, "found_rate": n_found / len(in_bin) if in_bin else math.nan,
                    "n_large": sum(lesion.large for lesion in in_bin),
                    "box_psnr": "", "box_ssim": "",
                })  # fmt: skip
            # per box: binned by its own longest side, so no grouping rule enters
            in_bin = [
                r for r in item_rows
                if size_bin(box_size[int(r["box_id"])], cfg.size_upper_bounds) == name
            ]  # fmt: skip
            n_found = sum(r["found"] == "True" for r in in_bin)
            out.append({
                "method": method, "speedup": speedup, "unit": "box", "connectivity": "",
                "size_bin": name, "n": len(in_bin), "n_found": n_found,
                "found_rate": n_found / len(in_bin) if in_bin else math.nan,
                "n_large": sum(box_size[int(r["box_id"])] > cfg.large_box_px for r in in_bin),
                "box_psnr": mean([float(r["box_psnr"]) for r in in_bin]),
                "box_ssim": mean([float(r["box_ssim"]) for r in in_bin]),
            })  # fmt: skip
    return out


def scores(rows: list[dict]) -> list[dict]:
    # slice scores repeat on every box row of a slice; each slice counts once
    slices: dict[tuple[str, float], dict[tuple[str, int], dict]] = {}
    for r in rows:
        slices.setdefault((r["method"], float(r["speedup"])), {})[r["file"], int(r["slice"])] = r
    out = []
    for (method, speedup), by_slice in sorted(slices.items()):
        values = list(by_slice.values())
        out.append(
            {"method": method, "speedup": speedup, "n_slices": len(values)}
            | {k: mean([float(r[k]) for r in values]) for k in SCORE_COLUMNS[3:]}
        )
    return out


def draw(found: list[dict], score: list[dict], method: str, connectivity: int, path: Path) -> None:
    fig = Figure(figsize=(13, 4), layout="constrained")
    ax_found, ax_psnr, ax_ssim = fig.subplots(1, 3)
    names = list(dict.fromkeys(r["size_bin"] for r in found))
    for i, name in enumerate(names):
        colour = f"C{i}"
        for unit, plural, style in (("lesion", "lesions", "-o"), ("box", "boxes", "--s")):
            points = [
                r for r in found
                if r["method"] == method and r["size_bin"] == name and r["unit"] == unit
                and r["connectivity"] in (connectivity, "")
            ]  # fmt: skip
            n = points[0]["n"] if points else 0
            xs = [r["speedup"] for r in points if not math.isnan(r["found_rate"])]
            ys = [r["found_rate"] for r in points if not math.isnan(r["found_rate"])]
            ax_found.plot(xs, ys, style, color=colour, label=f"{name} px, {plural} (n={n})")
    ax_found.set_ylim(-0.05, 1.05)
    ax_found.set_ylabel("found-rate")
    ax_found.set_title(f"found-rate by size group ({connectivity}-connected lesions)")
    ax_found.legend(fontsize="small")
    for ax, key, label in ((ax_psnr, "slice_psnr", "PSNR (dB)"), (ax_ssim, "slice_ssim", "SSIM")):
        # an exact copy (1x) has infinite psnr: kept in the table, left off the plot
        points = [r for r in score if r["method"] == method and math.isfinite(r[key])]
        ax.plot([r["speedup"] for r in points], [r[key] for r in points], "-o", color="black")
        ax.set_ylabel(label)
        ax.set_title(f"slice {label.split()[0]}, mean over slices")
    speedups = sorted({r["speedup"] for r in score if r["method"] == method})
    for ax in (ax_found, ax_psnr, ax_ssim):
        ax.set_xscale("log", base=2)
        ax.set_xticks(speedups, [f"{s:g}x" for s in speedups])
        ax.minorticks_off()
        ax.set_xlabel("speed-up (nominal)")
    fig.suptitle(f"{path.parent.parent.name}: {method}")
    fig.savefig(path, dpi=150)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml").lesions
    rows = read_results(run_dir / "results.csv")
    out_dir = run_dir / "report"
    out_dir.mkdir()  # fails if it exists: a report is never overwritten
    found = found_rates(rows, cfg)
    score = scores(rows)
    write_csv(out_dir / "found_rates.csv", FOUND_COLUMNS, found)
    write_csv(out_dir / "scores.csv", SCORE_COLUMNS, score)
    for method in dict.fromkeys(r["method"] for r in rows):
        draw(found, score, method, cfg.connectivity, out_dir / f"report_{method}.png")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report a run of the Lesion Preservation Test")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(report(args.run_dir))


if __name__ == "__main__":
    main()
