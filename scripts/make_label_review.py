"""사람이 채울 상자 검토표를 만든다. 판정은 채우지 않는다.

실행:  python scripts/make_label_review.py --bundle DATA_v001

입력:  <bundle>/dataset_work, <bundle>/audit/images.csv (--overlays 781로 실행한 결과)
출력:  <bundle>/review/label_review.csv, image_meta.csv

표본: 분할×접두 칸마다 무작위 1장, 객체가 가장 많은 1장, 박스가 가장 작은 1장을 고르고
      --target 장수까지 무작위로 채운다.
특이 사례: 한 변이 --small-px 이하인 박스가 있는 이미지와 353×336 이미지는 전부 넣고,
      테두리에 닿은 박스가 있는 이미지는 --border-sample 장을 넣는다.
목표 장수는 제안값이다. 오류가 반복되면 범위를 넓힌다.
"""
import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from data_common import (BORDER_PX, label_path, parse_name, read_csv, read_yolo, refuse_existing, write_csv)

FIELDS = ["image_id", "overlay_file", "reviewer", "decision", "notes", "selection_reason", "unlabeled_particles"]


def load(bundle):
    rows = read_csv(bundle / "audit" / "images.csv")
    images = []
    for r in rows:
        image_id, w, h = r["image_id"], int(r["width"]), int(r["height"])
        if not r.get("overlay_file"):
            raise SystemExit(f"{image_id}: 오버레이 없음. audit_dataset.py를 --overlays {len(rows)}로 다시 실행하세요.")
        px = [(x1 * w, y1 * h, x2 * w, y2 * h) for x1, y1, x2, y2 in read_yolo(label_path(bundle / "dataset_work", image_id))]
        prefix, number = parse_name(Path(image_id).name)
        images.append(dict(
            image_id=image_id, overlay_file=r["overlay_file"], split=r["original_split"], prefix=prefix, number=number,
            w=w, h=h, objects=len(px),
            median_area=statistics.median((b[2] - b[0]) * (b[3] - b[1]) for b in px),
            min_side=min(min(b[2] - b[0], b[3] - b[1]) for b in px),
            border=any(b[0] <= BORDER_PX or b[1] <= BORDER_PX or b[2] >= w - BORDER_PX or b[3] >= h - BORDER_PX for b in px)))
    return images


def select(images, target, small_px, border_n, seed):
    """이미지별 선정 이유를 돌려준다. 이유를 남겨 표본이 어떻게 뽑혔는지 추적한다."""
    rng = random.Random(seed)
    reasons = defaultdict(list)
    cells = defaultdict(list)
    for im in images:
        cells[(im["split"], im["prefix"])].append(im)

    def pick(pool, reason, key=None):
        pool = [x for x in pool if x["image_id"] not in reasons]
        if pool:
            x = rng.choice(pool) if key is None else min(pool, key=key)
            reasons[x["image_id"]].append(reason)

    for cell in sorted(cells):
        pool = sorted(cells[cell], key=lambda x: x["image_id"])
        pick(pool, "칸별_무작위")
        pick(pool, "칸별_객체최다", key=lambda x: (-x["objects"], x["image_id"]))
        pick(pool, "칸별_박스최소", key=lambda x: (x["median_area"], x["image_id"]))
    rest = sorted((x for x in images if x["image_id"] not in reasons), key=lambda x: x["image_id"])
    rng.shuffle(rest)
    for x in rest[:max(0, target - len(reasons))]:
        reasons[x["image_id"]].append("추가_무작위")

    for x in images:
        if x["min_side"] <= small_px:
            reasons[x["image_id"]].append(f"특이_변{small_px}px이하")
        if (x["w"], x["h"]) == (353, 336):
            reasons[x["image_id"]].append("특이_353x336")
    border = sorted((x for x in images if x["border"]), key=lambda x: x["image_id"])
    for x in rng.sample(border, min(border_n, len(border))):
        reasons[x["image_id"]].append("특이_테두리")
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--target", type=int, default=36)
    ap.add_argument("--small-px", type=int, default=5)
    ap.add_argument("--border-sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    bundle = a.bundle.resolve()
    out, meta = bundle / "review" / "label_review.csv", bundle / "review" / "image_meta.csv"
    refuse_existing(out, meta)

    images = load(bundle)
    reasons = select(images, a.target, a.small_px, a.border_sample, a.seed)
    rows = [dict(image_id=x["image_id"], overlay_file=x["overlay_file"], reviewer="", decision="", notes="",
                 selection_reason=";".join(reasons[x["image_id"]]), unlabeled_particles="")
            for x in sorted(images, key=lambda x: x["image_id"]) if x["image_id"] in reasons]
    write_csv(out, rows, FIELDS)
    write_csv(meta, [dict(image_id=x["image_id"], original_split=x["split"], filename_prefix=x["prefix"],
                          filename_number=x["number"], width=x["w"], height=x["h"], object_count=x["objects"])
                     for x in images],
              ["image_id", "original_split", "filename_prefix", "filename_number", "width", "height", "object_count"])

    special = sum(any(r.startswith("특이") for r in v) for v in reasons.values())
    print(f"검토 대상 {len(rows)}장 (특이 사례를 포함한 이미지 {special}장). decision, reviewer는 사람이 채웁니다.")
    print("decision: ok/issue/uncertain, unlabeled_particles: yes/no/unsure")


if __name__ == "__main__":
    main()
