"""Rebuild the feature3 YOLO dataset and R/G splits from the checked-in inputs."""
import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from tools.data_common import parse_name, read_csv, write_csv

EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parent
TOOLS = EXPERIMENT / "tools"
PENDING = {
    ("none", 1), ("none", 110), ("a", 84), ("b", 122),
    ("b", 84), ("c", 1), ("f", 200), ("f", 94),
}
D_GROUPS = {
    1: "G_D_1_28", 28: "G_D_1_28",
    12: "G_D_12_30", 30: "G_D_12_30",
    31: "G_D_31_32", 32: "G_D_31_32",
    39: "G_D_39_40", 40: "G_D_39_40",
}


def build_groups(out, images):
    ids = {Path(row["image_id"]).name: row["image_id"] for row in images}
    with ZipFile(REPO / "step2_split_n_build_datasets.zip") as archive:
        with io.TextIOWrapper(archive.open("groups_demo.csv"), encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
    names = [row["image_id"] for row in rows]
    if len(ids) != len(images) or len(set(names)) != len(names) or set(names) != set(ids):
        raise ValueError("feature2 groups and audited images must match one-to-one.")
    pending = {parse_name(row["image_id"]) for row in rows if row["group_id"] == "G_ISOLATE_PENDING_POOL"}
    if pending != PENDING:
        raise ValueError("feature2 pending set differs from the reviewed eight images.")

    changes = []
    for row in rows:
        name = row["image_id"]
        prefix, number = parse_name(name)
        old_group = row["group_id"]
        row["image_id"] = ids[name]
        if (prefix, number) in PENDING:
            row["group_id"] = f"solo_{Path(name).stem}"
            row["reviewer"] = "Codex"
            row["evidence"] = "원본 입자 배치와 라벨을 비교한 뒤 사용자 요청에 따라 독립 그룹으로 편입"
        elif prefix == "d" and number in D_GROUPS:
            row["group_id"] = D_GROUPS[number]
            if row["group_id"] != old_group:
                row["reviewer"] = "Codex"
                row["evidence"] = "feature2 중복 문서의 d--12 / d--30 그룹과 일치하도록 정확한 파일 번호로 매핑"
        if row["group_id"] != old_group:
            changes.append({
                "image_id": row["image_id"], "before": old_group, "after": row["group_id"],
                "reviewer": row["reviewer"], "reason": row["evidence"],
            })

    rows.sort(key=lambda row: row["image_id"])
    write_csv(out / "groups.csv", rows, ["image_id", "group_id", "reviewer", "evidence"])
    write_csv(out / "group_changes.csv", changes, ["image_id", "before", "after", "reviewer", "reason"])


def run(script, *args):
    subprocess.run([sys.executable, str(TOOLS / script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for a reproducible rebuild")
    out = parser.parse_args().out.resolve()
    if out.exists():
        parser.error("--out must be a new directory")
    if not out.parent.is_dir():
        parser.error("--out parent must exist")

    with TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary) / "build"
        bundle = stage / "data"
        dataset = bundle / "dataset_work"
        audit = Path(temporary) / "audit"
        run("build_dataset_work.py", "--source", REPO, "--bundle", bundle,
            "--approved-by", "feature3 build requested by user")
        run("audit_dataset.py", "--root", dataset, "--out", audit, "--overlays", 0)
        build_groups(stage, read_csv(audit / "images.csv"))
        # The audit CSVs with machine-specific image paths are temporary.
        run("make_splits.py", "--audit", audit, "--groups", stage / "groups.csv",
            "--classes", dataset / "classes.txt", "--out", stage / "splits", "--review-confirmed")
        (audit / "summary.json").replace(stage / "audit_summary.json")
        (audit / "issues.csv").replace(stage / "audit_issues.csv")
        stage.rename(out)


if __name__ == "__main__":
    main()
