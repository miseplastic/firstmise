"""DATA 묶음을 팀원 2에게 넘기기 전에 점검한다. 묶음의 기존 파일은 수정하지 않는다.

실행:  python scripts/check_data_bundle.py --bundle DATA_v001
       python scripts/check_data_bundle.py --bundle DATA_v001 --toolkit guideline/toolkit --rerun-audit
       python scripts/check_data_bundle.py --bundle DATA_v001 --open-items 미해결목록.csv

--rerun-audit: 작업본을 임시 경로로 복사해 audit를 다시 돌리고 결과를 비교한다.
--open-items:  미판정, uncertain, 미검토 후보, label issue 목록을 저장한다. 묶음 안에 둘 경로도 쓸 수 있다.

종료 코드: FAIL이 있으면 1, 없으면 0. 통과는 인계 가능하다는 뜻일 뿐 그룹이나 학습 승인이 아니다.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from data_common import LABEL_DECISIONS, PAIR_DECISIONS, parse_name, read_csv, sha256_file, write_csv

REQUIRED = [
    "dataset_work/classes.txt", "dataset_work/images", "dataset_work/labels",
    "audit/summary.json", "audit/images.csv", "audit/boxes.csv", "audit/issues.csv", "audit/overlays",
    "audit/exact_duplicates.csv", "audit/near_candidates.csv",
    "review/label_review.csv", "review/pair_review.csv", "review/change_log.csv", "review/file_mapping.csv",
    "review/source_manifest.csv", "review/image_meta.csv", "review/image_coverage.csv",
    "review/pair_candidates_all.csv", "review/pair_images", "review/duplicate_search_notes.md",
    "source_notes.md", "handoff.json",
]
STATUSES = {"READY_FOR_REVIEW", "REVIEW_NEEDED"}
AI_REVIEWER = re.compile(r"^(ai|claude|gpt|chatgpt|gemini|llm)([\s_\-(]|$)", re.I)
MIN_LABEL_REVIEWS = 20


class Report:
    def __init__(self):
        self.lines = []

    def add(self, level, text):
        self.lines.append((level, text))

    def check(self, ok, text, level="FAIL"):
        self.add("PASS" if ok else level, text)
        return ok


def bad_reviewer(name):
    return not name.strip() or bool(AI_REVIEWER.match(name.strip()))


def check_structure(b, rep):
    missing = [p for p in REQUIRED if not (b / p).exists()]
    rep.check(not missing, f"필수 파일 {len(REQUIRED)}개 확인" + (f": 없음 {missing}" if missing else ""))
    return not missing


def check_dataset(b, rep):
    work = b / "dataset_work"
    images = sorted(p.relative_to(work / "images").as_posix() for p in (work / "images").rglob("*.jpg"))
    labels = {p.relative_to(work / "labels").with_suffix("").as_posix() for p in (work / "labels").rglob("*.txt")}
    stems = {str(Path(i).with_suffix("")) .replace("\\", "/") for i in images}
    rep.check(stems == labels, f"이미지 {len(images)}장과 라벨 {len(labels)}개 대응")
    rows = read_csv(b / "audit" / "images.csv")
    ids = {r["image_id"] for r in rows}
    rep.check(ids == set(images), f"audit의 이미지 목록과 작업본 일치 ({len(ids)}장)")
    rep.check(not any(r.get("is_demo") == "true" for r in rows), "연습 자료 표시 없음")
    rep.check(not (work / "SYNTHETIC_DEMO.txt").exists(), "연습 자료 표식 파일 없음")

    errors = [r for r in read_csv(b / "audit" / "issues.csv") if r["severity"] == "ERROR"]
    rep.check(not errors, f"audit ERROR {len(errors)}건", level="WARN")

    hashes = {r["original_relative_path"]: r["sha256"] for r in read_csv(b / "review" / "source_manifest.csv")}
    bad, seen = [], 0
    for r in read_csv(b / "review" / "file_mapping.csv"):
        if r["working_relative_path"].startswith("images/"):
            seen += 1
            if hashes.get(r["original_relative_path"]) != sha256_file(work / r["working_relative_path"]):
                bad.append(r["working_relative_path"])
    rep.check(not bad and seen == len(images), f"작업본 이미지와 원본 해시 대조 {seen}장, 불일치 {len(bad)}장", level="WARN")
    return ids


def check_labels(b, ids, rep):
    rows = read_csv(b / "review" / "label_review.csv")
    unknown = [r["image_id"] for r in rows if r["image_id"] not in ids]
    rep.check(not unknown, f"label_review ID {len(rows)}건 확인" + (f": 없는 ID {unknown[:3]}" if unknown else ""))
    rep.check(all((b / "audit" / r["overlay_file"]).is_file() for r in rows), "label_review 오버레이 파일 확인")
    done = [r for r in rows if r["decision"].strip()]
    invalid = [r["image_id"] for r in done if r["decision"].strip() not in LABEL_DECISIONS]
    rep.check(not invalid, f"label_review 판정 값 확인" + (f": 잘못된 값 {invalid[:3]}" if invalid else ""))
    rep.check(len(done) >= MIN_LABEL_REVIEWS, f"상자 검토 {len(done)}장 (최소 {MIN_LABEL_REVIEWS}장)")
    rep.check(len(done) == len(rows), f"label_review 미판정 {len(rows) - len(done)}장", level="WARN")
    who = [r["image_id"] for r in done if bad_reviewer(r["reviewer"])]
    rep.check(not who, f"label_review 검토자 확인" + (f": 사람 이름 없음 {len(who)}건" if who else ""))
    cover = Counter(parse_name(Path(r["image_id"]).name)[0] for r in done)
    expected = {parse_name(Path(i).name)[0] for i in ids}
    rep.check(expected <= set(cover), f"상자 검토 접두 범위 {dict(cover)}", level="WARN")
    issues = [r for r in done if r["decision"].strip() == "issue"]
    changes = read_csv(b / "review" / "change_log.csv")
    logged = {r["image_id"] for r in changes}
    open_issue = [r["image_id"] for r in issues if r["image_id"] not in logged]
    rep.check(not open_issue, f"issue {len(issues)}건 중 조치 기록 없는 {len(open_issue)}건", level="WARN")
    marks = Counter(r.get("unlabeled_particles", "").strip() or "미기록" for r in rows)
    rep.add("INFO", f"라벨 없는 입자 표시: {dict(marks)}")
    return rows


def check_pairs(b, ids, rep):
    rows = read_csv(b / "review" / "pair_review.csv")
    unknown = [r for r in rows if r["image_a"] not in ids or r["image_b"] not in ids]
    rep.check(not unknown, f"pair_review ID {len(rows)}건 확인" + (f": 없는 ID {len(unknown)}건" if unknown else ""))
    done = [r for r in rows if r["decision"].strip()]
    invalid = [r for r in done if r["decision"].strip() not in PAIR_DECISIONS]
    rep.check(not invalid, "pair_review 판정 값 확인" + (f": 잘못된 값 {len(invalid)}건" if invalid else ""))
    rep.check(len(done) == len(rows), f"pair_review 판정 {len(done)}쌍, 미판정 {len(rows) - len(done)}쌍", level="WARN")
    who = [r for r in done if bad_reviewer(r["reviewer"])]
    rep.check(not who, "pair_review 검토자 확인" + (f": 사람 이름 없음 {len(who)}건" if who else ""))
    noev = [r for r in done if not r["evidence"].strip()]
    rep.check(not noev, "pair_review 근거 확인" + (f": 근거 없음 {len(noev)}건" if noev else ""))

    seen = {}
    conflicts = []
    for r in done:
        k = tuple(sorted((r["image_a"], r["image_b"])))
        if k in seen and seen[k] != r["decision"].strip():
            conflicts.append(k)
        seen[k] = r["decision"].strip()
    rep.check(not conflicts, f"같은 쌍의 판정 충돌 {len(conflicts)}건")

    groups = {}
    for r in read_csv(b / "audit" / "exact_duplicates.csv"):
        groups.setdefault(r["pixel_sha256"], []).append(r["image_id"])
    unseen = [g for g in groups.values() if any(tuple(sorted((g[0], x))) not in seen for x in g[1:])]
    rep.check(not unseen, f"완전 중복 {len(groups)}그룹 시각 검토, 미검토 {len(unseen)}그룹")
    rep.add("INFO", f"판정 분포: {dict(Counter(r['decision'].strip() for r in done))}")
    return rows


def check_change_log(b, ids, rep):
    rows = read_csv(b / "review" / "change_log.csv")
    required = {"date", "image_id", "issue", "action", "reason", "reviewer", "version"}
    with (b / "review" / "change_log.csv").open(encoding="utf-8-sig") as f:
        header = set(f.readline().strip().split(","))
    rep.check(required <= header, "change_log 열 확인")
    bad = [r["image_id"] for r in rows if not all(r.get(k, "").strip() for k in required)]
    rep.check(not bad, f"change_log {len(rows)}건의 필수 값 확인")


def check_notes(b, rep):
    blanks = [ln.strip() for ln in (b / "source_notes.md").read_text(encoding="utf-8").splitlines()
              if re.match(r"^\s*-\s*[^:]+:\s*$", ln)]
    rep.check(not blanks, f"source_notes 빈 항목 {len(blanks)}개" + (f": {blanks[:3]}" if blanks else ""))


def check_handoff(b, rep):
    try:
        h = json.loads((b / "handoff.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        rep.add("FAIL", f"handoff.json을 읽을 수 없음: {e}")
        return
    rep.check(h.get("stage") == "DATA" and str(h.get("version", "")).startswith("DATA_v"), "handoff 단계와 버전 확인")
    rep.check(h.get("status") in STATUSES, f"handoff 상태 {h.get('status')}. {sorted(STATUSES)}만 허용")
    rep.check(bool(str(h.get("reviewer", "")).strip()) and not bad_reviewer(str(h.get("reviewer", ""))), "handoff 검토자 확인")
    rep.check(bool(str(h.get("created_at", "")).strip()), "handoff 작성일 확인")
    files = h.get("files") or []
    missing = [f for f in files if not (b / f).exists()]
    rep.check(bool(files) and not missing, f"handoff 파일 {len(files)}개 확인" + (f": 없음 {missing}" if missing else ""))
    listed = set(files)
    unlisted = [p for p in REQUIRED if p != "handoff.json" and p not in listed and not any(p.startswith(f.rstrip("/") + "/") for f in listed)]
    rep.check(not unlisted, "필수 파일이 handoff에 포함됨" + (f": 빠짐 {unlisted}" if unlisted else ""))
    unresolved = json.dumps(h.get("unresolved_issues") or [], ensure_ascii=False).lower()
    has_test = (b / "dataset_work" / "images" / "test").exists()
    rep.check(has_test or "test" in unresolved, "test 분할 부재가 unresolved_issues에 기록됨")
    if h.get("status") == "READY_FOR_REVIEW" and any(lv == "WARN" for lv, _ in rep.lines):
        rep.add("WARN", "경고가 남아 있습니다. REVIEW_NEEDED가 맞는지 확인하세요")


def rerun_audit(b, toolkit, rep):
    script = toolkit / "audit_dataset.py"
    if not script.is_file():
        rep.add("FAIL", f"audit_dataset.py 없음: {script}")
        return
    with tempfile.TemporaryDirectory() as td:
        copy = Path(td) / "다른 경로" / "dataset_work"
        shutil.copytree(b / "dataset_work", copy)
        out = Path(td) / "audit_copy"
        run = subprocess.run([sys.executable, str(script), "--root", str(copy), "--out", str(out), "--near", "--overlays", "0"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not (out / "images.csv").exists():
            rep.add("FAIL", f"audit 재실행 실패 (종료 코드 {run.returncode})")
            return
        key = lambda rows: {r["image_id"]: (r["pixel_sha256"], r["dhash"], r["object_count"]) for r in rows}
        same = key(read_csv(out / "images.csv")) == key(read_csv(b / "audit" / "images.csv"))
        near = lambda p: len(read_csv(p / "near_candidates.csv"))
        rep.check(same and near(out) == near(b / "audit"), "다른 경로에서 audit 재실행 결과 일치")


def write_open_items(b, path, labels, pairs):
    items = []
    for r in labels:
        d = r["decision"].strip()
        if d != "ok":
            items.append(dict(kind="label", image_a=r["image_id"], image_b="", status=d or "미판정", notes=r["notes"]))
    reviewed = {tuple(sorted((r["image_a"], r["image_b"]))) for r in pairs if r["decision"].strip()}
    for r in pairs:
        d = r["decision"].strip()
        if d != "same" and d != "different":
            items.append(dict(kind="pair", image_a=r["image_a"], image_b=r["image_b"], status=d or "미판정", notes=r["notes"]))
    listed = {tuple(sorted((r["image_a"], r["image_b"]))) for r in pairs}
    for r in read_csv(b / "review" / "pair_candidates_all.csv"):
        k = tuple(sorted((r["image_a"], r["image_b"])))
        if k not in reviewed and k not in listed:
            items.append(dict(kind="pair", image_a=r["image_a"], image_b=r["image_b"], status="검토표 밖 후보", notes=""))
    write_csv(path, items, ["kind", "image_a", "image_b", "status", "notes"])
    return Counter(i["status"] for i in items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--toolkit", type=Path, default=Path("guideline/toolkit"))
    ap.add_argument("--rerun-audit", action="store_true")
    ap.add_argument("--open-items", type=Path)
    a = ap.parse_args()
    b = a.bundle.resolve()
    rep = Report()
    labels, pairs = None, None
    if check_structure(b, rep):
        ids = check_dataset(b, rep)
        labels = check_labels(b, ids, rep)
        pairs = check_pairs(b, ids, rep)
        check_change_log(b, ids, rep)
        check_notes(b, rep)
        check_handoff(b, rep)
        if a.rerun_audit:
            rerun_audit(b, a.toolkit.resolve(), rep)
    for level, text in rep.lines:
        print(f"[{level}] {text}")
    if a.open_items and labels is not None:
        print(f"미해결 목록 저장: {a.open_items} {dict(write_open_items(b, a.open_items.resolve(), labels, pairs))}")
    count = Counter(lv for lv, _ in rep.lines)
    print(f"\nFAIL {count['FAIL']}, WARN {count['WARN']}, PASS {count['PASS']}. 통과가 그룹이나 학습 승인을 뜻하지는 않습니다.")
    raise SystemExit(1 if count["FAIL"] else 0)


if __name__ == "__main__":
    main()
