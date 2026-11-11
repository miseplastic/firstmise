"""Validate a split and run YOLOv5 v7.0 with a machine-local data.yaml."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
RUNS = tuple(f"{condition}_{repeat}" for condition in ("R", "G") for repeat in (1, 2, 3))


def validate(root, run):
    directory = root / "splits" / run
    images = root / "data" / "dataset_work" / "images"
    labels = images.with_name("labels")
    config = json.loads((directory / "data.yaml").read_text(encoding="utf-8"))
    classes = (images.parent / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
    if config["names"] != classes:
        raise ValueError(f"{run}: data.yaml classes differ from classes.txt")
    condition, repeat = run.split("_")
    with (root / "splits" / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if (row["condition"], row["repeat"]) == (condition, repeat)]
    seen_images, seen_groups = set(), set()
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        expected = {row["image_id"] for row in split_rows}
        groups = {row["group_id"] for row in split_rows}
        listing = (directory / config[split]).resolve()
        entries = listing.read_text(encoding="utf-8").splitlines()
        ids = set()
        for entry in entries:
            image = (listing.parent / entry).resolve()
            image_id = image.relative_to(images).as_posix()
            label = (labels / image_id).with_suffix(".txt")
            if not entry.startswith("./") or not image.is_file() or not label.is_file():
                raise ValueError(f"{run}/{split}: invalid image or label path: {entry}")
            ids.add(image_id)
        if not ids or len(ids) != len(entries) or len(expected) != len(split_rows) or ids != expected:
            raise ValueError(f"{run}/{split}: image list and manifest differ")
        if ids & seen_images or (condition == "G" and groups & seen_groups):
            raise ValueError(f"{run}/{split}: image or G group crosses splits")
        seen_images.update(ids)
        seen_groups.update(groups)
        config[split] = str(listing)
        print(f"{run}/{split}: {len(ids)} images, {len(groups)} groups")
    seeds = {int(row["train_seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError(f"{run}: expected exactly one training seed")
    return config, seeds.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS, action="append")
    parser.add_argument("--check", action="store_true", help="Check data files without starting training")
    parser.add_argument("--yolov5", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=ROOT / ".runtime" / "runs")
    args = parser.parse_args()
    gate = json.loads((ROOT / "splits" / "review_gate.json").read_text(encoding="utf-8"))
    if hashlib.sha256((ROOT / "groups.csv").read_bytes()).hexdigest() != gate["grouping_sha256"]:
        parser.error("Group table changed after split generation. Rebuild the splits.")
    if args.check:
        for run in args.run or RUNS:
            validate(ROOT, run)
        return
    if not all((args.run, args.yolov5, args.weights, args.epochs, args.batch_size)):
        parser.error("training needs --run, --yolov5, --weights, --epochs and --batch-size")

    repo, weights, project = args.yolov5.resolve(), args.weights.resolve(), args.project.resolve()
    if not (repo / "train.py").is_file() or not weights.is_file():
        parser.error("Provide a YOLOv5 checkout and a local weights file.")
    if len(set(args.run)) != len(args.run):
        parser.error("Specify each --run once.")
    for run in args.run:
        if (project / run).exists():
            parser.error(f"Run already exists: {project / run}. Choose a new --project.")
    prepared = [(run, *validate(ROOT, run)) for run in args.run]
    for run, config, seed in prepared:
        runtime = project / "configs" / run
        runtime.mkdir(parents=True, exist_ok=True)
        data = runtime / "data.yaml"
        data.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, "-u", str(repo / "train.py"), "--data", str(data), "--weights", str(weights),
                   "--seed", str(seed), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--imgsz", str(args.imgsz), "--workers", str(args.workers), "--device", args.device,
                   "--project", str(project), "--name", run]
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
