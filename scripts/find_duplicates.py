"""중복 후보를 찾고 사람이 채울 검토표를 만든다. 판정은 채우지 않는다.

실행:  python scripts/find_duplicates.py --bundle DATA_v001

입력:  <bundle>/dataset_work, <bundle>/audit/{images,exact_duplicates,near_candidates}.csv
출력:  <bundle>/review/pair_review.csv            사람이 채울 검토표
       <bundle>/review/pair_candidates_all.csv    신호별 상위 후보 전체
       <bundle>/review/pair_images/               쌍 비교 이미지
       <bundle>/review/image_coverage.csv         이미지별 탐색 범위
       <bundle>/review/duplicate_search_notes.md  방법, 탐지 성능, 한계

신호 (유사도일 뿐 같은 원본의 증거가 아니다):
  dhash   툴킷과 같은 dHash
  layout  박스 배치 점수. 6가지 변환
  ncc     박스 영역 안의 고역통과 상관. 6가지 변환, 위치 이동 없음
  shift   박스 영역 지도의 교차상관 최댓값. 위치 이동을 허용하고 축소 배율을 일부 시험

탐지 성능: 실제 이미지를 변형한 사본을 넣어, 원본과의 쌍이 후보에 드는 비율을 잰다.
개수와 임계값은 탐색용 제안값이다. 검토 범위는 팀원 2와 합의한다.
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy import fft
from scipy.ndimage import gaussian_filter

from data_common import label_path, parse_name, read_csv, read_yolo, refuse_existing, write_csv

SIZE = 128        # 고역통과 영상 크기
MAP_SIZE = 64     # 교차상관 지도 크기
SHRINK = (1.0, 0.9, 0.8)  # 크롭 후 확대된 이미지를 맞춰 보기 위한 축소 배율
TRANSFORMS = ("id", "fliph", "flipv", "r90", "r180", "r270")
SIGNALS = ("dhash", "layout", "ncc", "shift")


# ---------------------------------------------------------------- 특징
def apply_array(a, t):
    """(..., H, W) 배열에 변환을 적용한다."""
    if t == "id":
        return a
    if t == "fliph":
        return a[..., ::-1]
    if t == "flipv":
        return a[..., ::-1, :]
    return np.rot90(a, {"r90": 1, "r180": 2, "r270": 3}[t], axes=(-2, -1))


def apply_boxes(b, t):
    """정규화 박스 (x1, y1, x2, y2) 배열에 apply_array와 같은 변환을 적용한다."""
    x1, y1, x2, y2 = b.T
    return {"id": b,
            "fliph": np.stack([1 - x2, y1, 1 - x1, y2], 1),
            "flipv": np.stack([x1, 1 - y2, x2, 1 - y1], 1),
            "r90": np.stack([y1, 1 - x2, y2, 1 - x1], 1),
            "r180": np.stack([1 - x2, 1 - y2, 1 - x1, 1 - y1], 1),
            "r270": np.stack([1 - y2, x1, 1 - y1, x2], 1)}[t]


def dhash_bits(img):
    d = np.asarray(img.convert("L").resize((9, 8)), dtype=np.int16)
    return (d[:, :-1] > d[:, 1:]).ravel()


def box_mask(boxes, size):
    m = np.zeros((size, size), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        m[int(y1 * size):max(int(y1 * size) + 1, int(np.ceil(y2 * size))),
          int(x1 * size):max(int(x1 * size) + 1, int(np.ceil(x2 * size)))] = 1
    return m


class Features:
    def __init__(self, items):
        """items: (RGB 이미지, 정규화 박스 배열) 목록"""
        hp, masks, maps, h, f, self.boxes = [], [], [], [], [], []
        for img, boxes in items:
            g = np.asarray(img.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
            x = g - gaussian_filter(g, 8)
            m = box_mask(boxes, SIZE)
            hp.append(x)
            masks.append(m)
            p = (x * m).reshape(MAP_SIZE, 2, MAP_SIZE, 2).mean((1, 3))
            maps.append(p / max(np.linalg.norm(p), 1e-6))
            h.append(dhash_bits(img))
            f.append(dhash_bits(img.transpose(Image.FLIP_LEFT_RIGHT)))
            self.boxes.append(boxes)
        self.hp, self.masks, self.maps = np.stack(hp), np.stack(masks), np.stack(maps)
        self.h, self.f = np.stack(h).astype(np.float32), np.stack(f).astype(np.float32)

    def __len__(self):
        return len(self.boxes)


# ---------------------------------------------------------------- 신호
def hamming(a, b):
    return a @ (1 - b).T + (1 - a) @ b.T


def dhash_scores(a, b):
    """a의 원본, 좌우 반전과 b의 원본을 비교한다. 툴킷 audit_dataset.py와 같은 방식이다."""
    return np.minimum(hamming(a.h, b.h), hamming(a.f, b.h))


def ncc_one_way(a, b):
    """a의 박스 영역 안에서 a와 변환된 b의 상관을 구한다."""
    x = a.hp.reshape(len(a), -1)
    m = a.masks.reshape(len(a), -1)
    cnt = m.sum(1)[:, None]
    mx = m * x
    sa, saa = mx.sum(1)[:, None], (mx * x).sum(1)[:, None]
    va = saa - sa ** 2 / cnt
    best, arg = None, None
    for k, t in enumerate(TRANSFORMS):
        y = apply_array(b.hp, t).reshape(len(b), -1)
        sb, sbb, sab = m @ y.T, m @ (y * y).T, mx @ y.T
        cov = sab - sa * sb / cnt
        vb = sbb - sb ** 2 / cnt
        s = cov / np.sqrt(np.maximum(va * vb, 1e-6))
        best, arg = (s, np.full(s.shape, k)) if best is None else (np.maximum(best, s), np.where(s > best, k, arg))
    return best, arg


def shrink_map(p, scale):
    if scale == 1.0:
        return p
    n = int(round(MAP_SIZE * scale))
    out = np.zeros_like(p)
    o = (MAP_SIZE - n) // 2
    for i, im in enumerate(p):
        out[i, o:o + n, o:o + n] = np.asarray(Image.fromarray(im, mode="F").resize((n, n), Image.BILINEAR))
        out[i] /= max(np.linalg.norm(out[i]), 1e-6)
    return out


def shift_one_way(a, b):
    """a와 변환·축소된 b의 원형 교차상관 최댓값을 구한다."""
    fa = fft.rfft2(a.maps, workers=-1)
    best, arg = np.zeros((len(a), len(b)), np.float32), np.zeros((len(a), len(b)), int)
    for k, t in enumerate(TRANSFORMS):
        for scale in SHRINK:
            fb = np.conj(fft.rfft2(shrink_map(apply_array(b.maps, t).copy(), scale), workers=-1))
            for i in range(len(a)):
                s = fft.irfft2(fa[i] * fb, s=(MAP_SIZE, MAP_SIZE), workers=-1).max((1, 2))
                better = s > best[i]
                best[i][better], arg[i][better] = s[better], k
    return best, arg


def layout_score(a, b):
    """정규화 박스 집합의 양방향 최대 IoU 평균. REPORT_01과 같은 점수에 회전을 더했다."""
    best, kind = 0.0, 0
    for k, t in enumerate(TRANSFORMS):
        c = apply_boxes(b, t)
        ix = np.clip(np.minimum(a[:, None, 2], c[None, :, 2]) - np.maximum(a[:, None, 0], c[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(a[:, None, 3], c[None, :, 3]) - np.maximum(a[:, None, 1], c[None, :, 1]), 0, None)
        inter = ix * iy
        aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        cc = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
        iou = inter / (aa[:, None] + cc[None, :] - inter)
        s = (iou.max(1).sum() + iou.max(0).sum()) / (len(a) + len(c))
        if s > best:
            best, kind = s, k
    return best, kind


def layout_matrix(a, b, same):
    score, arg = np.zeros((len(a), len(b)), np.float32), np.zeros((len(a), len(b)), int)
    for i, x in enumerate(a.boxes):
        for j in range(i + 1 if same else 0, len(b)):
            y = b.boxes[j]
            # 박스 수가 크게 다르면 건너뛴다. 크롭으로 박스가 줄어든 사본은 놓칠 수 있다
            if abs(len(x) - len(y)) <= max(2, 0.3 * max(len(x), len(y))):
                score[i, j], arg[i, j] = layout_score(x, y)
    if same:
        score += score.T
        arg += arg.T
    return score, arg


def all_scores(a, b=None):
    """쌍 점수 행렬을 만든다. b가 없으면 a 안의 모든 쌍을 계산한다."""
    same = b is None
    b = a if same else b
    dh = dhash_scores(a, b)
    if same:
        # 앞선 이미지를 기준으로 계산하는 툴킷과 맞춘다
        dh = np.triu(dh, 1) + np.triu(dh, 1).T
    else:
        dh = np.minimum(dh, dhash_scores(b, a).T)
    n_ab, n_arg = ncc_one_way(a, b)
    s_ab, s_arg = shift_one_way(a, b)
    n_ba = n_ab.T if same else ncc_one_way(b, a)[0].T
    s_ba = s_ab.T if same else shift_one_way(b, a)[0].T
    lay, lay_arg = layout_matrix(a, b, same)
    return dict(dhash=dh, layout=lay, ncc=(n_ab + n_ba) / 2, shift=np.maximum(s_ab, s_ba),
                layout_arg=lay_arg, ncc_arg=n_arg, shift_arg=s_arg)


# ---------------------------------------------------------------- 탐지 성능
def crop_variant(img, boxes, frac, ox, oy):
    """이미지를 잘라 원래 크기로 되돌린다. 남은 부분이 30% 미만인 박스는 뺀다."""
    w, h = img.size
    x0, y0 = int(ox * (1 - frac) * w), int(oy * (1 - frac) * h)
    x1, y1 = x0 + int(frac * w), y0 + int(frac * h)
    out = img.crop((x0, y0, x1, y1)).resize((w, h), Image.BILINEAR)
    nx0, ny0, nx1, ny1 = x0 / w, y0 / h, x1 / w, y1 / h
    kept = []
    for bx1, by1, bx2, by2 in boxes:
        cx1, cy1, cx2, cy2 = max(bx1, nx0), max(by1, ny0), min(bx2, nx1), min(by2, ny1)
        if cx2 > cx1 and cy2 > cy1 and (cx2 - cx1) * (cy2 - cy1) >= 0.3 * (bx2 - bx1) * (by2 - by1):
            kept.append(((cx1 - nx0) / frac, (cy1 - ny0) / frac, (cx2 - nx0) / frac, (cy2 - ny0) / frac))
    return out, np.array(kept) if kept else None


def reencode(img, quality):
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_variants(img, boxes):
    pil = {"fliph": Image.FLIP_LEFT_RIGHT, "flipv": Image.FLIP_TOP_BOTTOM, "r90": Image.ROTATE_90,
           "r180": Image.ROTATE_180, "r270": Image.ROTATE_270}
    out = {"재인코딩(q70)": (reencode(img, 70), boxes),
           "밝기+15%": (ImageEnhance.Brightness(img).enhance(1.15), boxes)}
    names = {"fliph": "좌우 반전", "flipv": "상하 반전", "r90": "90도 회전", "r180": "180도 회전", "r270": "270도 회전"}
    for t, p in pil.items():
        out[names[t]] = (img.transpose(p), apply_boxes(boxes, t))
    # 정사각형으로 줄여 계산하므로 90도 회전 사본은 비정사각 이미지의 비율이 조금 달라진다
    for name, args in (("중앙 크롭 90%", (0.9, 0.5, 0.5)), ("중앙 크롭 80%", (0.8, 0.5, 0.5)),
                       ("한쪽 크롭 80%", (0.8, 0.0, 0.0))):
        im, b = crop_variant(img, boxes, *args)
        if b is not None:
            out[name] = (im, b)
    return out


def injection_test(images, feats, scores, thresholds, inject_n, seed, distance):
    """표본의 변형 사본과 원본의 쌍이 후보 임계값을 넘는 비율을 잰다."""
    rng = random.Random(seed)
    by_prefix = defaultdict(list)
    for i, im in enumerate(images):
        by_prefix[im["prefix"]].append(i)
    picks = []
    for p in sorted(by_prefix):
        picks += rng.sample(by_prefix[p], min(inject_n, len(by_prefix[p])))
    items, meta = [], []
    for i in picks:
        for name, (im, b) in make_variants(images[i]["img"], images[i]["boxes"]).items():
            items.append((im, b))
            meta.append((name, i))
    s = all_scores(Features(items), feats)
    rows = defaultdict(lambda: Counter())
    for v, (name, i) in enumerate(meta):
        hit = {k: (s[k][v, i] <= distance if k == "dhash" else s[k][v, i] >= thresholds[k]) for k in SIGNALS}
        r = rows[name]
        r["n"] += 1
        r["any"] += any(hit.values())
        for k in SIGNALS:
            r[k] += bool(hit[k])
    return rows, len(picks)


# ---------------------------------------------------------------- 후보와 검토표
def pair_ranks(scores, n, per_signal):
    """신호별 순위와 신호별 상위 후보 집합을 만든다."""
    iu = np.triu_indices(n, 1)
    ranks, thresholds, cand = {}, {}, set()
    for k in SIGNALS:
        v = scores[k][iu]
        order = np.argsort(v if k == "dhash" else -v, kind="stable")
        r = np.empty(len(v), dtype=int)
        r[order] = np.arange(len(v))
        ranks[k] = r
        thresholds[k] = float(v[order[per_signal - 1]])
    for k in SIGNALS:
        cand |= {(int(iu[0][x]), int(iu[1][x])) for x in np.where(ranks[k] < per_signal)[0]}
    return iu, ranks, thresholds, cand


def largest_component(pairs, n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    return max(Counter(find(i) for i in range(n)).values())


def draw_panel(img, boxes, size, title):
    w, h = img.size
    s = min(size / w, size / h)
    im = img.resize((int(w * s), int(h * s)))
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in boxes:
        d.rectangle((x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height), outline=(255, 60, 60))
    canvas = Image.new("RGB", (size, size + 18), "white")
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((3, 3), title, fill="black")
    return canvas


PIL_T = {"fliph": Image.FLIP_LEFT_RIGHT, "flipv": Image.FLIP_TOP_BOTTOM, "r90": Image.ROTATE_90,
         "r180": Image.ROTATE_180, "r270": Image.ROTATE_270}


def comparison(a, b, t, path):
    """A, B, 변환한 B를 나란히 둔다."""
    tb = b["img"] if t == "id" else b["img"].transpose(PIL_T[t])
    panels = [draw_panel(a["img"], a["boxes"], 420, "A " + a["image_id"][:48]),
              draw_panel(b["img"], b["boxes"], 420, "B " + b["image_id"][:48]),
              draw_panel(tb, apply_boxes(b["boxes"], t), 420, f"B with transform: {t}")]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 8, panels[0].height), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 4
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def select_review(ranks, iu, cand, top, boundary, rand, outside, per_signal, seed, exact):
    """검토 대상을 고른다. 판정 결과를 부풀리지 않도록 상위, 경계, 무작위를 함께 넣는다."""
    rng = random.Random(seed)
    chosen = {p: "완전중복" for p in exact}
    index = {}
    for k in SIGNALS:
        index[k] = np.argsort(ranks[k])
    pos = 0
    start = len(chosen)
    while len(chosen) - start < top:
        for k in SIGNALS:
            x = index[k][pos]
            p = (int(iu[0][x]), int(iu[1][x]))
            if p not in chosen and len(chosen) - start < top:
                chosen[p] = "신호상위"
        pos += 1
    best = {}
    for k in SIGNALS:
        for x in np.where(ranks[k] < per_signal)[0]:
            p = (int(iu[0][x]), int(iu[1][x]))
            best[p] = min(best.get(p, per_signal), int(ranks[k][x]))
    upper = max(pos, 1)
    edge = sorted(p for p, r in best.items() if p not in chosen and upper <= r and r < upper + per_signal // 4)
    for p in rng.sample(edge, min(boundary, len(edge))):
        chosen[p] = "경계"
    rest = sorted(p for p in cand if p not in chosen)
    for p in rng.sample(rest, min(rand, len(rest))):
        chosen[p] = "후보_무작위"
    n = int(iu[1].max()) + 1
    picked = 0
    while picked < outside:
        a, b = sorted(rng.sample(range(n), 2))
        if (a, b) not in cand and (a, b) not in chosen:
            chosen[(a, b)] = "비후보_무작위"
            picked += 1
    return chosen


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--per-signal", type=int, default=500, help="신호별 후보 수")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--boundary", type=int, default=20)
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--outside", type=int, default=10)
    ap.add_argument("--inject-per-prefix", type=int, default=4)
    ap.add_argument("--distance", type=int, default=5, help="툴킷 dHash 후보 거리")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    bundle = a.bundle.resolve()
    work, review, audit = bundle / "dataset_work", bundle / "review", bundle / "audit"
    refuse_existing(review / "pair_review.csv", review / "pair_candidates_all.csv", review / "image_coverage.csv",
                    review / "duplicate_search_notes.md", review / "pair_images")

    rows = read_csv(audit / "images.csv")
    images = []
    for r in rows:
        with Image.open(work / "images" / r["image_id"]) as im:
            img = im.convert("RGB")
        boxes = np.array(read_yolo(label_path(work, r["image_id"])))
        images.append(dict(image_id=r["image_id"], img=img, boxes=boxes, split=r["original_split"],
                           prefix=parse_name(Path(r["image_id"]).name)[0]))
    ids = {im["image_id"]: i for i, im in enumerate(images)}
    n = len(images)
    print(f"이미지 {n}장 특징 계산", flush=True)
    feats = Features([(im["img"], im["boxes"]) for im in images])
    print("전체 쌍 점수 계산 (몇 분 걸립니다)", flush=True)
    scores = all_scores(feats)

    iu, ranks, thresholds, cand = pair_ranks(scores, n, a.per_signal)
    # audit는 픽셀이 같은 쌍을 near_candidates에서 뺀다. 그런 쌍은 완전 중복으로 따로 다룬다
    pixel = [r["pixel_sha256"] for r in rows]
    dh_pairs = [(int(i), int(j)) for i, j in zip(*iu) if scores["dhash"][i, j] <= a.distance and pixel[i] != pixel[j]]
    audit_near = len(read_csv(audit / "near_candidates.csv"))
    if len(dh_pairs) != audit_near:
        raise SystemExit(f"dHash 후보 수가 audit({audit_near})와 다릅니다({len(dh_pairs)}). 같은 audit 결과인지 확인하세요.")
    cand |= set(dh_pairs)

    exact_groups = defaultdict(list)
    for r in read_csv(audit / "exact_duplicates.csv"):
        exact_groups[r["pixel_sha256"]].append(ids[r["image_id"]])
    exact = [tuple(sorted((g[0], x))) for g in exact_groups.values() for x in g[1:]]
    cand |= set(exact)

    chosen = select_review(ranks, iu, cand, a.top, a.boundary, a.random, a.outside, a.per_signal, a.seed, exact)

    def best_transform(i, j):
        k = min(("layout", "ncc", "shift"), key=lambda s: ranks[s][pos_of(i, j)])
        return TRANSFORMS[int(scores[k + "_arg"][i, j])]

    offsets = np.cumsum(np.r_[0, n - 1 - np.arange(n - 1)])

    def pos_of(i, j):
        return int(offsets[i] + j - i - 1)

    def score_row(i, j):
        return dict(image_a=images[i]["image_id"], image_b=images[j]["image_id"],
                    split_a=images[i]["split"], split_b=images[j]["split"],
                    dhash_distance=int(scores["dhash"][i, j]), layout_score=round(float(scores["layout"][i, j]), 3),
                    masked_ncc=round(float(scores["ncc"][i, j]), 3), shift_peak=round(float(scores["shift"][i, j]), 3),
                    best_transform=best_transform(i, j))

    score_fields = ["split_a", "split_b", "dhash_distance", "layout_score", "masked_ncc", "shift_peak", "best_transform"]
    review_rows = []
    order = {"완전중복": 0, "신호상위": 1, "경계": 2, "후보_무작위": 3, "비후보_무작위": 4}
    tiers = sorted(chosen.items(), key=lambda kv: (order[kv[1]], kv[0]))
    # 순서가 판정에 영향을 주지 않도록 파일 번호를 섞는다. 검토 구분은 표에 남긴다
    numbers = list(range(1, len(tiers) + 1))
    random.Random(a.seed).shuffle(numbers)
    for ((i, j), tier), no in zip(tiers, numbers):
        rel = f"pair_images/{no:04d}.png"
        row = score_row(i, j)
        comparison(images[i], images[j], row["best_transform"], review / rel)
        row.update(decision="", reviewer="", evidence="", notes="", review_tier=tier, comparison_file=rel, _no=no)
        review_rows.append(row)
    review_rows.sort(key=lambda r: r["_no"])
    write_csv(review / "pair_review.csv", review_rows,
              ["image_a", "image_b", "decision", "reviewer", "evidence", "notes", "review_tier", "comparison_file"] + score_fields)
    write_csv(review / "pair_candidates_all.csv",
              [dict(score_row(i, j), in_pair_review=(i, j) in chosen) for i, j in sorted(cand)],
              ["image_a", "image_b"] + score_fields + ["in_pair_review"])

    def nearest(mat, i, low):
        v = mat[i].copy().astype(float)
        v[i] = np.inf if low else -np.inf
        j = int(v.argmin() if low else v.argmax())
        return images[j]["image_id"], round(float(v[j]), 3)

    cand_count, review_count = Counter(), Counter()
    for i, j in cand:
        cand_count[i] += 1
        cand_count[j] += 1
    for i, j in chosen:
        review_count[i] += 1
        review_count[j] += 1
    coverage = []
    for i, im in enumerate(images):
        row = dict(image_id=im["image_id"], search_methods="dhash;layout;masked_ncc;shift_peak",
                   candidate_pairs=cand_count[i], pairs_in_review=review_count[i])
        for key, mat, low in (("dhash", scores["dhash"], True), ("layout", scores["layout"], False),
                              ("ncc", scores["ncc"], False), ("shift", scores["shift"], False)):
            row[f"nearest_{key}"], row[f"score_{key}"] = nearest(mat, i, low)
        coverage.append(row)
    write_csv(review / "image_coverage.csv", coverage,
              ["image_id", "search_methods", "candidate_pairs", "pairs_in_review"]
              + [f"{p}_{k}" for k in ("dhash", "layout", "ncc", "shift") for p in ("nearest", "score")])

    print("탐지 성능 측정", flush=True)
    inj, picks = injection_test(images, feats, scores, thresholds, a.inject_per_prefix, a.seed, a.distance)
    table = ["| 변형 | 사본 수 | 후보에 포함 | dHash | 박스 배치 | 박스 영역 상관 | 위치 이동 상관 |", "|---|---|---|---|---|---|---|"]
    for name, r in inj.items():
        pct = lambda k: f"{100 * r[k] / r['n']:.0f}%"
        table.append(f"| {name} | {r['n']} | {pct('any')} | {pct('dhash')} | {pct('layout')} | {pct('ncc')} | {pct('shift')} |")

    no_review = sum(1 for i in range(n) if review_count[i] == 0)
    notes = f"""# 중복 후보 탐색 기록

이 문서는 `scripts/find_duplicates.py`가 만들었다. 후보는 같은 원본의 증거가 아니며, 판정은 `pair_review.csv`에 사람이 남긴다.

## 방법

| 신호 | 내용 | 후보 기준 |
|---|---|---|
| dHash | 툴킷과 같은 방식, 좌우 반전 포함 | 거리 {a.distance} 이하 |
| 박스 배치 | 양방향 최대 IoU 평균, 6가지 변환 | 상위 {a.per_signal}쌍 (점수 {thresholds['layout']:.3f} 이상) |
| 박스 영역 상관 | 박스 영역 안의 고역통과 상관, 6가지 변환 | 상위 {a.per_signal}쌍 (점수 {thresholds['ncc']:.3f} 이상) |
| 위치 이동 상관 | 박스 영역 지도의 교차상관, 6가지 변환과 배율 {SHRINK} | 상위 {a.per_signal}쌍 (점수 {thresholds['shift']:.3f} 이상) |

임계값은 이 데이터의 순위에서 정한 탐색용 값이다. 보편 기준이 아니다.

## 후보 수

- 전체 쌍 {len(iu[0]):,}개 중 후보 {len(cand):,}쌍, 검토표 {len(chosen)}쌍 (구분별 {dict(Counter(chosen.values()))})
- 검토표에 한 번도 들지 않은 이미지 {no_review}장
- 후보 전체는 `pair_candidates_all.csv`, 이미지별 범위는 `image_coverage.csv`에 있다
- 검토 범위는 제안값이며 팀원 2와 합의한다. 미검토 후보는 중복이 아니라는 뜻이 아니다

## 탐지 성능

접두별 최대 {a.inject_per_prefix}장, 합계 {picks}장의 실제 이미지를 변형한 사본과 원본의 쌍이 후보 기준을 넘은 비율이다. 표본이 작아 대략적인 값이다.

{chr(10).join(table)}

이 표는 파일 안의 합성 변형에 대한 값이다. 실제 데이터의 변형 방식이 다르면 탐지 비율도 달라진다.

## 알려진 한계

- 전역 dHash와 고역통과 상관은 배경의 영향을 받는다.
- 입력 박스에 의존하는 신호는 라벨이 틀리거나 빠지면 약해진다.
- 시험하지 않은 크롭 비율, 원근 변화, 라벨을 함께 바꾼 사본은 놓칠 수 있다.
- 조각 추가나 재배치처럼 보이는 사례는 시료 ID가 없어 같은 대상인지 확인할 수 없다. `uncertain`으로 남긴다.

## 팀원 2 주의

dHash 후보를 연결해 그룹으로 묶지 않는다. 거리 {a.distance} 이하를 연결하면 최대 {largest_component(dh_pairs, n)}장의 묶음이 생긴다.
"""
    (review / "duplicate_search_notes.md").write_text(notes, encoding="utf-8")
    print(f"후보 {len(cand)}쌍, 검토표 {len(chosen)}쌍. decision, reviewer, evidence는 사람이 채웁니다.")


if __name__ == "__main__":
    main()
