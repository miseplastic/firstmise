import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

CONDITIONS = ('0', '50A', '50B', '100')
SEEDS = (101, 102, 103)
YOLO_COMMIT = '9777cd00e86382dfa5a12e6244ec4f609bc4774c'
WEIGHTS_SHA = '8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee'


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boxes(path):
    result = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = list(map(float, line.split()))
        require(len(row) == 5 and all(math.isfinite(v) for v in row), f'Invalid label: {path}')
        c, x, y, w, h = row
        require(c == 0 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1,
                f'Invalid single-class YOLO box: {path}')
        require(x-w/2 >= -1e-6 and y-h/2 >= -1e-6 and x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6,
                f'Box outside image: {path}')
        result.append(row)
    require(len({tuple(r) for r in result}) == len(result), f'Duplicate GT rows: {path}')
    return result


def check_bundle(directory):
    directory = Path(directory).resolve()
    lock = read_json(directory/'bundle.lock.json')
    for name, sha in lock['files'].items():
        p = directory/name
        require(p.is_file() and digest(p) == sha, f'Bundle changed/missing: {name}')
    plan = read_json(directory/'plan.json')
    require(plan['schema_version'] == 1 and plan['conditions'] == list(CONDITIONS), 'Unsupported bundle')
    return plan, digest(directory/'bundle.lock.json')


def check_yolo(directory, weights=None):
    directory = Path(directory).resolve()
    commit = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == YOLO_COMMIT, f'Use verified server YOLO commit {YOLO_COMMIT}; found {commit}')
    # Untracked runtime files do not change model code; tracked modifications do.
    dirty = subprocess.check_output(['git', '-C', str(directory), 'diff', 'HEAD', '--name-only'], text=True)
    require(not dirty.strip(), 'YOLO tracked files modified; inspect before running')
    if weights:
        require(digest(weights) == WEIGHTS_SHA, 'Initial weights differ from the verified server file')
    return commit


def load_settings(path):
    # Training uses YOLO's installed PyYAML; CPU-only analysis does not need it.
    try:
        return read_json(path)
    except json.JSONDecodeError:
        pass
    import yaml
    with Path(path).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def source_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
