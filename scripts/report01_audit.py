"""1차 데이터 감사 (docs/REPORT_01.md 의 수치를 재현한다).

실행:  python scripts/report01_audit.py            # 프로젝트 루트 자동 인식
       python scripts/report01_audit.py --root PATH

입력:  train/, valid/ 의 jpg 와 _annotations.csv, artifacts/data_inventory.csv
출력:  artifacts/ 아래
       report01_image_audit.csv   이미지별 SHA-256, 디코딩 결과, 실제·주석 크기, 밝기
       exact_duplicates.csv       SHA-256 일치 파일 (현재 0건)
       phash_candidates.csv       pHash 최소 거리 <= 4 인 쌍
       phash_nearest.csv          이미지별 같은/다른 분할 최근접 이미지와 거리
       box_layout_candidates.csv  박스 배치 점수 >= 0.5 인 쌍
       report01_candidate_pairs.png, report01_prefix_samples.png
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

SPLITS = ("train", "valid")
TRANSFORMS = {
    "id": None,
    "fliph": Image.FLIP_LEFT_RIGHT,
    "flipv": Image.FLIP_TOP_BOTTOM,
    "r90": Image.ROTATE_90,
    "r180": Image.ROTATE_180,
    "r270": Image.ROTATE_270,
}
PHASH_SAVE_MAX = 4        # phash_candidates.csv 에 저장할 최대 거리
PHASH_REPORT = (0, 2, 4, 8)
LAYOUT_SAVE_MIN = 0.5


# ---------------------------------------------------------------- 공통
def load(root: Path):
    ann = pd.concat(
        [pd.read_csv(root / s / "_annotations.csv").assign(split=s) for s in SPLITS],
        ignore_index=True,
    )
    inv = pd.read_csv(root / "artifacts" / "data_inventory.csv")
    inv["prefix"] = inv["prefix"].fillna("")
    return ann, inv


def phash(img: Image.Image, size: int = 8, factor: int = 4) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((size * factor,) * 2, Image.LANCZOS), dtype=float)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:size, :size].flatten()
    return d > np.median(d[1:])


# ---------------------------------------------------------------- 1. 무결성 + 해시
def image_audit(root: Path, inv: pd.DataFrame):
    rows, hashes = [], {}
    for r in inv.itertuples():
        path = root / r.image_path
        data = path.read_bytes()
        row = dict(filename=r.filename, split=r.split, prefix=r.prefix, sample_id=r.sample_id,
                   sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                   decode_ok=True, err="", real_w=None, real_h=None,
                   ann_w=r.width, ann_h=r.height, mean_gray=np.nan, std_gray=np.nan)
        try:
            with Image.open(path) as im:
                im.verify()
            im = Image.open(path)
            im.load()
            row["real_w"], row["real_h"] = im.size
            gray = np.asarray(im.convert("L"), dtype=float)
            row["mean_gray"], row["std_gray"] = gray.mean(), gray.std()
            hashes[r.filename] = {k: phash(im if t is None else im.transpose(t))
                                  for k, t in TRANSFORMS.items()}
        except Exception as e:  # noqa: BLE001
            row["decode_ok"], row["err"] = False, str(e)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["size_match"] = (df.real_w == df.ann_w) & (df.real_h == df.ann_h)
    return df, hashes


def exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (df[df.duplicated("sha256", keep=False)]
            .sort_values("sha256")[["sha256", "filename", "split", "prefix", "sample_id"]])


def phash_analysis(df: pd.DataFrame, hashes: dict):
    meta = df[df.decode_ok].set_index("filename")
    names = list(meta.index)
    keys = list(TRANSFORMS)
    base = np.array([hashes[n]["id"] for n in names])
    dist = np.stack([
        (base[:, None, :] != np.array([hashes[n][k] for n in names])[None, :, :]).sum(-1)
        for k in keys
    ])
    dmin, darg = dist.min(0), dist.argmin(0)
    split = meta.split.values
    iu = np.triu_indices(len(names), 1)

    sel = dmin[iu] <= PHASH_SAVE_MAX
    i, j = iu[0][sel], iu[1][sel]
    cand = pd.DataFrame(dict(
        a=np.array(names)[i], b=np.array(names)[j], split_a=split[i], split_b=split[j],
        prefix_a=meta.prefix.values[i], prefix_b=meta.prefix.values[j],
        id_a=meta.sample_id.values[i], id_b=meta.sample_id.values[j],
        dist_identity=dist[0][i, j], dist_min=dmin[i, j],
        best_transform=np.array(keys)[darg[i, j]],
    ))

    summary = []
    sa, sb = split[iu[0]], split[iu[1]]
    for t in PHASH_REPORT:
        m = dmin[iu] <= t
        summary.append(dict(max_dist=t, pairs=m.sum(),
                            train_train=(m & (sa == "train") & (sb == "train")).sum(),
                            valid_valid=(m & (sa == "valid") & (sb == "valid")).sum(),
                            train_valid=(m & (sa != sb)).sum()))

    nearest = []
    for k, n in enumerate(names):
        other = split != split[k]
        same = split == split[k]
        same[k] = False
        io, is_ = np.where(other)[0], np.where(same)[0]
        nearest.append(dict(
            filename=n, split=split[k], prefix=meta.prefix.iloc[k], sample_id=meta.sample_id.iloc[k],
            nn_cross_dist=dmin[k, io].min(), nn_cross=names[io[dmin[k, io].argmin()]],
            nn_same_dist=dmin[k, is_].min(), nn_same=names[is_[dmin[k, is_].argmin()]],
        ))
    nearest = pd.DataFrame(nearest)

    # 파일명 번호가 같은 train-valid 쌍의 거리
    idx = {n: k for k, n in enumerate(names)}
    tr = meta[meta.split == "train"]
    same_num = []
    for n, r in meta[meta.split == "valid"].iterrows():
        c = tr[tr.sample_id == r.sample_id]
        if len(c):
            same_num.append(dict(min_dist=min(dmin[idx[n], idx[m]] for m in c.index),
                                 same_prefix=bool((c.prefix == r.prefix).any())))
    same_num = pd.DataFrame(same_num)
    cross_median = np.median(dmin[np.ix_(split == "valid", split == "train")])
    return cand, pd.DataFrame(summary), nearest, same_num, np.percentile(dmin[iu], [1, 5, 50]), cross_median


# ---------------------------------------------------------------- 2. 박스 통계
def box_stats(ann: pd.DataFrame, inv: pd.DataFrame):
    a = ann.merge(inv[["filename", "prefix"]], on="filename")
    a["pfx"] = a.prefix.replace("", "(none)")
    a["bw"], a["bh"] = a.xmax - a.xmin, a.ymax - a.ymin
    a["area"] = a.bw * a.bh
    a["size"] = np.sqrt(a.area)
    a["coco"] = pd.cut(a.area, [0, 32**2, 96**2, np.inf], labels=["small", "medium", "large"])
    a["border"] = ((a.xmin <= 1) | (a.ymin <= 1) | (a.xmax >= a.width - 1) | (a.ymax >= a.height - 1))
    per_img = a.groupby(["split", "pfx", "filename"]).size().rename("boxes").reset_index()
    print("\n[박스 크기 sqrt(area) 분위수]")
    print(a.groupby("split")["size"].describe(percentiles=[.25, .5, .75]).round(1))
    print("\n[COCO 크기 비율]")
    print(pd.crosstab(a.split, a.coco, normalize="index").round(3))
    print(pd.crosstab([a.split, a.pfx], a.coco, normalize="index").round(2))
    print("\n[이미지당 박스 수]")
    print(per_img.groupby("split").boxes.describe(percentiles=[.25, .5, .75]))
    print(per_img.groupby(["split", "pfx"]).boxes.agg(["count", "sum", "median"]))
    print("\n[접두별 박스 크기 중앙값]")
    print(a.groupby(["split", "pfx"])["size"].median().unstack().round(1))
    print("\n[테두리에 닿은 박스 비율]")
    print(a.groupby("split").border.mean().round(3))
    print("\n[가로세로비 중앙값]", (a.bw / a.bh).groupby(a.split).median().round(2).to_dict())
    print("[같은 이미지 내 좌표가 완전히 같은 박스]",
          a.duplicated(["filename", "xmin", "ymin", "xmax", "ymax"]).sum())
    return a


def _iou(A, B):
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter)


def _flip(b, kind):
    x1, y1, x2, y2 = b.T
    return {"id": b,
            "fliph": np.stack([1 - x2, y1, 1 - x1, y2], 1),
            "flipv": np.stack([x1, 1 - y2, x2, 1 - y1], 1),
            "r180": np.stack([1 - x2, 1 - y2, 1 - x1, 1 - y1], 1)}[kind]


def layout_candidates(a: pd.DataFrame) -> pd.DataFrame:
    """정규화 박스 집합의 양방향 최대 IoU 평균. 박스 수 차이가 큰 쌍은 건너뛴다."""
    boxes = {f: (g[["xmin", "ymin", "xmax", "ymax"]].values
                 / np.array([g.width.iloc[0], g.height.iloc[0]] * 2))
             for f, g in a.groupby("filename")}
    split = a.groupby("filename").split.first()
    names = list(boxes)
    out = []
    for i, fa in enumerate(names):
        A = boxes[fa]
        for fb in names[i + 1:]:
            B = boxes[fb]
            if abs(len(A) - len(B)) > max(2, 0.3 * max(len(A), len(B))):
                continue
            best, kind = 0.0, None
            for k in ("id", "fliph", "flipv", "r180"):
                m = _iou(A, _flip(B, k))
                s = (m.max(1).sum() + m.max(0).sum()) / (len(A) + len(B))
                if s > best:
                    best, kind = s, k
            if best >= LAYOUT_SAVE_MIN:
                out.append(dict(a=fa, b=fb, split_a=split[fa], split_b=split[fb], na=len(A), nb=len(B),
                                layout_score=round(best, 3), transform=kind))
    return pd.DataFrame(out, columns=["a", "b", "split_a", "split_b", "na", "nb", "layout_score", "transform"])


# ---------------------------------------------------------------- 3. 그림
def montage(root, items, cols, cell, path, labels=None):
    from PIL import ImageDraw
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cell * cols, cell * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (split, fname) in enumerate(items):
        r, c = divmod(k, cols)
        im = Image.open(root / split / fname).convert("RGB").resize((cell - 4, cell - 4))
        canvas.paste(im, (c * cell + 2, r * cell + 2))
    for r, text in enumerate(labels or []):
        draw.text((6, r * cell + 6), text, fill="yellow")
    canvas.save(path)


def figures(root, audit, phash_cand, layout):
    art = root / "artifacts"
    z = phash_cand[phash_cand.dist_min == 0]
    pairs = list(z[z.split_a != z.split_b][["a", "b", "split_a", "split_b"]].values[:3])
    pairs += list(z[z.split_a == z.split_b][["a", "b", "split_a", "split_b"]].values[:1])
    pairs += list(layout.sort_values("layout_score", ascending=False)[["a", "b", "split_a", "split_b"]].values[:2])
    items = [x for a, b, sa, sb in pairs for x in ((sa, a), (sb, b))]
    montage(root, items, 2, 300, art / "report01_candidate_pairs.png")

    items, labels = [], []
    for p in ["", "a", "b", "c", "d", "f"]:
        s = audit[audit.prefix == p].sample(4, random_state=1)
        items += list(zip(s.split, s.filename))
        labels.append("prefix=" + (p or "none"))
    montage(root, items, 4, 200, art / "report01_prefix_samples.png", labels)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = ap.parse_args().root.resolve()
    art = root / "artifacts"
    ann, inv = load(root)

    audit, hashes = image_audit(root, inv)
    audit.to_csv(art / "report01_image_audit.csv", index=False)
    print(f"[무결성] 이미지 {len(audit)}장, 디코딩 실패 {(~audit.decode_ok).sum()}, "
          f"크기 불일치 {(~audit.size_match).sum()}")
    print(audit.groupby(["real_w", "real_h"]).size().to_string())

    dup = exact_duplicates(audit)
    dup.to_csv(art / "exact_duplicates.csv", index=False)
    print(f"\n[SHA-256] 일치 파일 {len(dup)}개, 그룹 {dup.sha256.nunique()}개")

    cand, summary, nearest, same_num, q, cross_median = phash_analysis(audit, hashes)
    cand.to_csv(art / "phash_candidates.csv", index=False)
    nearest.to_csv(art / "phash_nearest.csv", index=False)
    print("\n[pHash 최소 거리별 쌍 수]")
    print(summary.to_string(index=False))
    print("전체 쌍 거리 분위수 1/5/50%:", q)
    v = nearest[nearest.split == "valid"]
    print("valid -> train 최근접 거리 중앙값:", v.nn_cross_dist.median(),
          "| <=4:", (v.nn_cross_dist <= 4).sum(), "| <=8:", (v.nn_cross_dist <= 8).sum(), f"/ {len(v)}")
    print("접두별 중앙값:", v.groupby("prefix").nn_cross_dist.median().to_dict())
    print(f"\n[파일명 번호] valid 중 train과 번호 공유 {len(same_num)}장, 접두까지 같은 경우 "
          f"{same_num.same_prefix.sum()}건, 거리 중앙값 {same_num.min_dist.median()} "
          f"(무작위 train-valid 쌍 {cross_median})")

    a = box_stats(ann, inv)
    print("\n[gray 중앙값]")
    print(audit.groupby(["split", "prefix"])[["mean_gray", "std_gray"]].median().round(1))

    layout = layout_candidates(a)
    layout.to_csv(art / "box_layout_candidates.csv", index=False)
    for t in (0.5, 0.6, 0.7):
        s = layout[layout.layout_score >= t]
        print(f"[박스 배치 >= {t}] {len(s)}쌍, train-valid {(s.split_a != s.split_b).sum()}쌍")

    figures(root, audit, cand, layout)
    print("\n완료:", art)


if __name__ == "__main__":
    main()
