"""DATA_v001 스크립트가 함께 쓰는 입출력 함수."""
import csv
import hashlib
import json
import re
from pathlib import Path

CLASS_NAME = "Microplastic"
SPLIT_MAP = {"train": "train", "valid": "val"}  # 제공 폴더명 -> 작업본 폴더명
LABEL_DECISIONS = {"ok", "issue", "uncertain"}
PAIR_DECISIONS = {"same", "different", "uncertain"}
BORDER_PX = 1  # REPORT_01과 같은 테두리 기준

_NAME_RE = re.compile(r"^(?:(?P<prefix>[a-z])--(?P<a>\d+)-|(?P<b>\d+))_jpg\.rf\.[0-9a-f]+\.jpg$")


def parse_name(name):
    """파일명에서 (접두, 번호)를 읽는다. 접두나 번호가 시료를 뜻한다는 근거는 없다."""
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"예상하지 못한 파일명: {name}")
    return m["prefix"] or "none", int(m["a"] or m["b"])


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_yolo(path):
    """YOLO 라벨을 (x1, y1, x2, y2) 정규화 좌표 목록으로 읽는다."""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def label_path(dataset_work, image_id):
    return (Path(dataset_work) / "labels" / image_id).with_suffix(".txt")


def refuse_existing(*paths):
    """사람 판정이 담길 수 있는 파일을 덮어쓰지 않는다."""
    found = [str(p) for p in paths if Path(p).exists()]
    if found:
        raise SystemExit("이미 있는 파일은 덮어쓰지 않습니다. 새 버전 폴더를 쓰세요: " + ", ".join(found))
