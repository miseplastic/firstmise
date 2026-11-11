"""제공 CSV 데이터로 툴킷용 작업본을 만든다. 원본은 읽기만 한다.

실행:  python scripts/build_dataset_work.py --source . --bundle DATA_v001 --approved-by 이름

--source: train/, valid/가 있는 폴더. 두 폴더만 읽는다.
--approved-by: 변환 방식을 승인한 사람. 지시문이 라벨 자동 변환을 금지하므로 필수다.

출력:  <bundle>/dataset_work/{classes.txt, images/{train,val}, labels/{train,val}}
       <bundle>/review/file_mapping.csv, source_manifest.csv, conversion_report.json

제공 분할에 test가 없으므로 images/test를 만들지 않는다. 공식 분할을 새로 만들지 않기 위해서다.
라벨 없는 이미지를 빈 배경으로 처리하지 않고 멈춘다.
입력을 검증한 뒤 임시 폴더에 만들고, 왕복 검증을 통과해야 dataset_work로 옮긴다.
"""
import argparse
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

from data_common import CLASS_NAME, SPLIT_MAP, read_csv, refuse_existing, sha256_file, write_csv, write_json

COLUMNS = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}


def load_source(source):
    """입력을 검증한다. 문제가 하나라도 있으면 아무것도 쓰지 않고 멈출 수 있게 목록을 돌려준다."""
    problems, images, manifest = [], [], []
    for split, work_split in SPLIT_MAP.items():
        folder = source / split
        files = sorted(p for p in folder.iterdir() if p.is_file())
        manifest += [dict(original_relative_path=p.relative_to(source).as_posix(), bytes=p.stat().st_size,
                          sha256=sha256_file(p)) for p in files]
        others = [p.name for p in files if p.suffix != ".jpg" and p.name != "_annotations.csv"]
        if others:
            problems.append(f"{split}: 예상하지 못한 파일 {others[:5]}")
        rows = read_csv(folder / "_annotations.csv")
        if rows and set(rows[0]) != COLUMNS:
            problems.append(f"{split}: CSV 열이 다름 {sorted(rows[0])}")
            continue
        by_file = defaultdict(list)
        for r in rows:
            by_file[r["filename"]].append(r)
        disk = {p.name for p in files if p.suffix == ".jpg"}
        for name in sorted(disk - by_file.keys()):
            problems.append(f"{split}/{name}: 라벨 행 없음. 빈 배경으로 처리하지 않음")
        for name in sorted(by_file.keys() - disk):
            problems.append(f"{split}/{name}: 이미지 파일 없음")
        for name in sorted(disk & by_file.keys()):
            with Image.open(folder / name) as im:
                im.load()
                w, h = im.size
            boxes = []
            for r in by_file[name]:
                if r["class"] != CLASS_NAME:
                    problems.append(f"{split}/{name}: 클래스 {r['class']}")
                x1, y1, x2, y2 = (int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                if (int(r["width"]), int(r["height"])) != (w, h):
                    problems.append(f"{split}/{name}: 주석 크기와 실제 크기 {w}x{h}가 다름")
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    problems.append(f"{split}/{name}: 잘못된 좌표 {(x1, y1, x2, y2)}")
                boxes.append((x1, y1, x2, y2))
            images.append(dict(split=split, work_split=work_split, name=name, w=w, h=h, boxes=boxes,
                               sha256=sha256_file(folder / name)))
    return images, manifest, problems


def to_line(box, w, h):
    x1, y1, x2, y2 = box
    return f"0 {(x1 + x2) / 2 / w:.8f} {(y1 + y2) / 2 / h:.8f} {(x2 - x1) / w:.8f} {(y2 - y1) / h:.8f}"


def from_line(line, w, h):
    _, cx, cy, bw, bh = line.split()
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return tuple(round(v) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--approved-by", required=True, help="변환 방식을 승인한 사람")
    a = ap.parse_args()
    source, bundle = a.source.resolve(), a.bundle.resolve()
    if not a.approved_by.strip():
        raise SystemExit("--approved-by에 승인자를 적어야 합니다.")
    inputs = [source / s for s in SPLIT_MAP]
    if bundle == source or any(bundle == p or p in bundle.parents for p in inputs):
        raise SystemExit("작업본은 원본 train, valid 폴더 밖에 만들어야 합니다.")
    work, review = bundle / "dataset_work", bundle / "review"
    refuse_existing(work, review / "file_mapping.csv", review / "source_manifest.csv")

    images, manifest, problems = load_source(source)
    if problems:
        print("\n".join(problems[:50]))
        raise SystemExit(f"입력 문제 {len(problems)}건. 아무것도 만들지 않았습니다.")

    tmp = bundle / "dataset_work.tmp"
    if tmp.exists():
        raise SystemExit(f"이전 실행의 임시 폴더를 확인하세요: {tmp}")
    mapping, mismatches = [], []
    for s in SPLIT_MAP.values():
        (tmp / "images" / s).mkdir(parents=True)
        (tmp / "labels" / s).mkdir(parents=True)
    (tmp / "classes.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")
    for im in images:
        rel_img = f"{im['work_split']}/{im['name']}"
        dst = tmp / "images" / rel_img
        shutil.copyfile(source / im["split"] / im["name"], dst)
        if sha256_file(dst) != im["sha256"]:
            mismatches.append(f"{rel_img}: 복사 후 해시가 다름")
        lbl = (tmp / "labels" / rel_img).with_suffix(".txt")
        lbl.write_text("\n".join(to_line(b, im["w"], im["h"]) for b in im["boxes"]) + "\n", encoding="utf-8")
        back = [from_line(x, im["w"], im["h"]) for x in lbl.read_text(encoding="utf-8").splitlines()]
        if sorted(back) != sorted(im["boxes"]):
            mismatches.append(f"{rel_img}: 왕복 좌표가 다름")
        reason = "변경 없음" if im["split"] == im["work_split"] else "폴더 이름을 valid에서 val로 변경"
        mapping.append(dict(original_relative_path=f"{im['split']}/{im['name']}",
                            working_relative_path=f"images/{rel_img}", reason=reason))
        mapping.append(dict(original_relative_path=f"{im['split']}/_annotations.csv",
                            working_relative_path=lbl.relative_to(tmp).as_posix(), reason="CSV 행을 YOLO txt로 변환"))
    if mismatches:
        print("\n".join(mismatches[:50]))
        raise SystemExit(f"검증 실패 {len(mismatches)}건. 임시 폴더를 확인하세요: {tmp}")
    tmp.rename(work)

    write_csv(review / "file_mapping.csv", mapping, ["original_relative_path", "working_relative_path", "reason"])
    write_csv(review / "source_manifest.csv", manifest, ["original_relative_path", "bytes", "sha256"])
    boxes = sum(len(i["boxes"]) for i in images)
    report = dict(date=date.today().isoformat(), approved_by=a.approved_by, images=len(images), boxes=boxes,
                  round_trip_mismatches=0, copy_hash_mismatches=0,
                  classes=[CLASS_NAME], split_map=SPLIT_MAP, test_split="없음. 제공 분할에 없어 만들지 않음",
                  box_convention="CSV의 xmin, ymin, xmax, ymax를 픽셀 모서리 좌표로 보고 변환함. 출처 문서로는 미확인")
    write_json(review / "conversion_report.json", report)
    print(f"이미지 {len(images)}장, 박스 {boxes}개 변환. 왕복 불일치 0건. 다음 단계: audit_dataset.py")


if __name__ == "__main__":
    main()
