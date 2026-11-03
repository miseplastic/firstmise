# 미세플라스틱 탐지의 데이터 분할 민감도 분석

## 연구 목적

유사·중복 이미지의 존재 여부를 감사하고, 이미지 단위 분할과 그룹 단위 분할에서 YOLOv5s의 성능을 비교한다.

## 연구 질문

- RQ1: 공식 train/valid 사이에 완전 중복 또는 동일 원본의 변형 이미지가 있는가?
- RQ2: 동일 원본 그룹을 분리하지 않는 분할에서 탐지 성능은 어떻게 달라지는가?

## 현재 단계 (2026-11-03)

데이터 감사 1차를 마쳤다. 결과는 [docs/REPORT_01.md](docs/REPORT_01.md)에 있다.

- SHA-256 완전 중복은 0건이다.
- 전역 pHash는 배경이 지배해 중복 판별에 쓸 수 없다. 거리가 0인 쌍도 서로 다른 장면이었다.
- valid의 66%가 `c` 접두라 train과 입자 분포가 다르다.

사람 검토, 그룹 확정, 분할 생성, 탐지 성능 실험은 아직 하지 않았다. 분할 비율, seed, 학습 설정도 미확정이다.

## 처음 보는 사람이 읽을 순서

1. [docs/REPORT_01.md](docs/REPORT_01.md): 지금까지 데이터에서 확인한 내용. 0절 요약부터 읽는다
2. [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md): 확정된 기준, 미확정 항목, 결정 기록
3. [docs/WORKLOG.md](docs/WORKLOG.md): 작업 목록(T01–T06)과 날짜별 기록

## 폴더 구조

```
miseplastic/
├─ README.md
├─ requirements.txt
├─ train/                     원본 데이터 (jpg 577장 + _annotations.csv)
├─ valid/                     원본 데이터 (jpg 204장 + _annotations.csv)
├─ docs/
│  ├─ REPORT_01.md            1차 데이터 분석 보고서
│  ├─ RESEARCH_PLAN.md        연구 기준·실험 설계·결정 기록
│  ├─ WORKLOG.md              작업 현황과 기록
│  └─ references/             참고 캡처 (데이터 점검 체크리스트, 그룹 판정 원칙)
├─ notebook/
│  └─ 01_data_inventory.ipynb 이미지 목록과 분포 그림 생성 (T01)
├─ scripts/
│  └─ report01_audit.py       무결성·SHA-256·pHash·박스 통계 감사 (REPORT_01)
└─ artifacts/                 위 노트북과 스크립트가 만든 결과 (재생성 가능)
```

### artifacts/ 파일

| 파일 | 만드는 곳 | 내용 |
|---|---|---|
| `data_inventory.csv` | notebook | 이미지별 분할, 크기, 박스 수, 접두, 번호 |
| `dataset_overview.png`, `sample_boxes.png` | notebook | 분포 그림, 박스 예시 |
| `report01_image_audit.csv` | script | 이미지별 SHA-256, 디코딩 결과, 실제 크기, 밝기 |
| `exact_duplicates.csv` | script | SHA-256 일치 파일 (현재 0건) |
| `phash_candidates.csv` | script | pHash 최소 거리 4 이하 쌍 |
| `phash_nearest.csv` | script | 이미지별 최근접 이미지와 거리 |
| `box_layout_candidates.csv` | script | 박스 배치 점수 0.5 이상 쌍 |
| `report01_candidate_pairs.png`, `report01_prefix_samples.png` | script | 후보 쌍 비교, 접두별 예시 |

## 실행 방법

Python 3.12 기준이다. Anaconda 환경이면 필요한 패키지가 이미 들어 있다.

```bash
pip install -r requirements.txt

# 1) 이미지 목록 생성 → artifacts/data_inventory.csv  (스크립트보다 먼저)
jupyter nbconvert --to notebook --execute --inplace notebook/01_data_inventory.ipynb

# 2) 1차 감사 → artifacts/report01_*, exact_duplicates.csv 등  (약 1–2분)
python scripts/report01_audit.py
```

학습, 평가 명령은 분할 기준이 확정된 뒤에 추가한다.

## 데이터 메모

- 주석 형식: CSV(`filename,width,height,class,xmin,ymin,xmax,ymax`), 클래스는 `Microplastic` 하나
- `test` 분할은 없다
- 출처 URL, 버전, 라이선스, 다운로드 날짜는 아직 기록되지 않았다. 공유 전에 이용 조건을 확인한다
