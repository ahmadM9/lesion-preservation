import argparse
import csv
import hashlib
import platform
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from lesion_preservation.config import Config, load_config
from lesion_preservation.data import load_slices
from lesion_preservation.judge import get_judge
from lesion_preservation.masks import apply_mask, effective_speedup, get_mask
from lesion_preservation.metrics import box_scores, nmse, psnr, ssim
from lesion_preservation.recon import get_reconstructor

COLUMNS = [
    "file", "slice", "box_id", "box_row", "box_col", "box_height", "box_width", "label",
    "method", "speedup", "effective_speedup", "mask", "centre_fraction", "seed", "judge",
    "found", "judge_score", "slice_psnr", "slice_ssim", "slice_nmse",
    "box_psnr", "box_ssim", "box_nmse", "recon_time",
]  # fmt: skip


def hms(seconds: float, millis: bool = False) -> str:
    whole = int(seconds)
    text = f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
    return text + f".{int((seconds - whole) * 1000):03d}" if millis else text


def code_version() -> str:
    # commit of the pipeline checkout, plus "dirty" when it has uncommitted changes
    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not a git checkout"
    return f"{commit} dirty" if status else commit


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def done_items(results: Path) -> set[tuple[str, int, float, str]]:
    # an item is one (file, slice, speed-up, method); its rows are written together
    if not results.exists():
        return set()
    with open(results, newline="") as f:
        return {
            (r["file"], int(r["slice"]), float(r["speedup"]), r["method"])
            for r in csv.DictReader(f)
        }


def start_run(config_path: Path, runs_dir: Path | None, resume: Path | None) -> Path:
    if resume is not None:
        run_dir = resume
        if load_config(run_dir / "config.yaml") != load_config(config_path):
            raise ValueError(f"{config_path} differs from the config in {run_dir}")
        recorded = (run_dir / "code_version.txt").read_text().strip()
        if recorded != code_version():
            raise ValueError(f"code changed since {run_dir} started: was {recorded}")
        return run_dir
    if runs_dir is None:
        raise ValueError("give --runs-dir for a new run, or --resume for an existing one")
    run_dir = runs_dir / f"{date.today().isoformat()}_{load_config(config_path).name}"
    run_dir.mkdir(parents=True)  # fails if the folder exists: runs are never overwritten
    shutil.copyfile(config_path, run_dir / "config.yaml")
    (run_dir / "code_version.txt").write_text(code_version() + "\n")
    return run_dir


def record_session(run_dir: Path, session: dict) -> None:
    # one entry per session, so a resumed run keeps the record of each part
    path = run_dir / "run_record.yaml"
    sessions = yaml.safe_load(path.read_text()) if path.exists() else []
    path.write_text(yaml.safe_dump([*sessions, session], sort_keys=False))


def run(
    config_path: str | Path,
    data_dir: str | Path,
    labels_csv: str | Path,
    runs_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    config_path, data_dir, labels_csv = Path(config_path), Path(data_dir), Path(labels_csv)
    cfg = load_config(config_path)
    run_dir = start_run(
        config_path,
        Path(runs_dir) if runs_dir is not None else None,
        Path(resume) if resume is not None else None,
    )
    started = datetime.now()
    clock = perf_counter()
    session = {
        "config_path": str(config_path),
        "labels_sha256": sha256(labels_csv),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        "start": started.isoformat(timespec="seconds"),
    }
    process(cfg, data_dir, labels_csv, run_dir)
    session["end"] = datetime.now().isoformat(timespec="seconds")
    session["runtime"] = hms(perf_counter() - clock)
    session["peak_gpu_memory_mb"] = (
        round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else "none"
    )
    record_session(run_dir, session)
    return run_dir


def process(cfg: Config, data_dir: Path, labels_csv: Path, run_dir: Path) -> None:
    results = run_dir / "results.csv"
    masks_path = run_dir / "masks.npz"
    done = done_items(results)
    masks = dict(np.load(masks_path)) if masks_path.exists() else {}
    make_mask = get_mask(cfg.mask.name)
    methods = {name: get_reconstructor(name) for name in cfg.methods}
    judge = get_judge(cfg.judge.name, **cfg.judge.params)

    with open(results, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
        for stem in cfg.files:
            for s in load_slices(data_dir / f"{stem}.h5", labels_csv):
                if not s.boxes:
                    continue
                data_range = float(s.reference.max())
                for speedup in cfg.speedups:
                    # one mask per width, speed-up and seed; every method gets the same one
                    key = f"{s.kspace.shape[-1]}_{speedup}_{cfg.mask.seed}"
                    if key not in masks:
                        masks[key] = make_mask(
                            s.kspace.shape[-1], speedup, cfg.mask.centre_fraction, cfg.mask.seed
                        )
                        np.savez(masks_path, **masks)
                    mask = masks[key]
                    masked = apply_mask(s.kspace, mask)
                    for name, method in methods.items():
                        if (stem, s.index, speedup, name) in done:
                            continue
                        tic = perf_counter()
                        image = method(masked, mask, s.reference.shape)
                        recon_time = perf_counter() - tic
                        slice_ssim, ssim_map = ssim(s.reference, image, data_range, cfg.ssim_window)
                        common = {
                            "file": stem, "slice": s.index, "method": name, "speedup": speedup,
                            "effective_speedup": effective_speedup(mask), "mask": cfg.mask.name,
                            "centre_fraction": cfg.mask.centre_fraction, "seed": cfg.mask.seed,
                            "judge": cfg.judge.name,
                            "slice_psnr": psnr(s.reference, image, data_range),
                            "slice_ssim": slice_ssim,
                            "slice_nmse": nmse(s.reference, image),
                            "recon_time": hms(recon_time, millis=True),
                        }  # fmt: skip
                        rows = []
                        for box, verdict in zip(s.boxes, judge(image, s.boxes), strict=True):
                            scores = box_scores(s.reference, image, ssim_map, box, data_range)
                            rows.append(
                                common
                                | {
                                    "box_id": box.box_id,
                                    "box_row": box.row,
                                    "box_col": box.col,
                                    "box_height": box.height,
                                    "box_width": box.width,
                                    "label": box.label,
                                    "found": verdict.found,
                                    "judge_score": verdict.score,
                                    "box_psnr": scores["psnr"],
                                    "box_ssim": scores["ssim"],
                                    "box_nmse": scores["nmse"],
                                }  # fmt: skip
                            )
                        # an item's rows go out together, so a crash loses one item at most
                        writer.writerows(rows)
                        f.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Lesion Preservation Test")
    parser.add_argument("config", type=Path)
    # machine paths: given here, not in the public config
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    print(run(args.config, args.data_dir, args.labels_csv, args.runs_dir, args.resume))


if __name__ == "__main__":
    main()
