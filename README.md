# 공개 미세플라스틱 이미지 데이터의 중복 감사와 YOLOv5s 분할별 성능 재검증

## 연구 목적

유사·중복 이미지의 존재 여부를 감사하고, 이미지 단위 분할과 그룹 단위 분할에서 YOLOv5s의 성능을 비교한다.

## 연구 질문

- RQ1: 공식 train/valid 사이에 완전 중복 또는 동일 원본의 변형 이미지가 있는가?
- RQ2: 동일 원본 그룹을 분리하지 않는 분할에서 탐지 성능은 어떻게 달라지는가?

## 현재 단계 (2026-11-06)

팀원 1의 `DATA_v001` 인계를 준비한다. 현재는 [로드맵](docs/ROADMAP.md) 0단계로, 팀 합의를 기다린다.

- SHA-256 완전 중복은 0건이다(파일 바이트 기준).
- 전역 pHash와 dHash는 배경의 영향이 커서 중복 판별 근거로 쓸 수 없다.
- valid의 66%가 `c` 접두라 train과 입자 분포가 다르다.
- 제공 분할에 test가 없어 O 목록의 처리 방침이 필요하다.

사람 검토, 그룹 확정, 분할 생성, 탐지 성능 실험은 아직 하지 않았다. 분할 비율, seed, 학습 설정은 제안값만 있고 확정되지 않았다.

## 역할

| 담당 | 책임 |
|---|---|
| 학습 담당 | 학교 서버, 환경, GPU 학습, 가중치 보존 |
| 팀원 1 | 데이터 확보, 기초 검사, 상자 검토, 중복 1차 판정 |
| 팀원 2 | 그룹 확정, 분할, 공정성 검수 |
| 팀원 3 | 독립 평가, 지표 집계, 오류 분석, 문서 취합 |

## 처음 보는 사람이 읽을 순서

1. [docs/ROADMAP.md](docs/ROADMAP.md): 인계까지의 단계와 완료 기준
2. [docs/TEAM_AGENDA.md](docs/TEAM_AGENDA.md): 팀이 정해야 할 안건과 받아야 할 입력
3. [docs/REPORT_01.md](docs/REPORT_01.md): 지금까지 데이터에서 확인한 내용. 0절 요약부터 읽는다
4. [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md): 확정된 기준, 미확정 항목, 결정 기록
5. [docs/WORKLOG.md](docs/WORKLOG.md): 작업 목록(T01–T06)과 날짜별 기록

## 폴더 구조

```
firstmise/
├─ README.md
├─ requirements.txt
├─ guideline/                 팀원 1 매뉴얼, 툴킷, 템플릿 (git 제외)
├─ toss to team2/             팀원 2 매뉴얼, 툴킷, 템플릿 (git 제외)
├─ train/                     원본 데이터 (jpg 577장 + _annotations.csv)
├─ valid/                     원본 데이터 (jpg 204장 + _annotations.csv)
├─ docs/
│  ├─ ROADMAP.md              DATA_v001 인계까지의 로드맵
│  ├─ TEAM_AGENDA.md          팀 합의 안건과 요청 입력
│  ├─ REPORT_01.md            1차 데이터 분석 보고서
│  ├─ RESEARCH_PLAN.md        연구 기준·실험 설계·결정 기록
│  ├─ WORKLOG.md              작업 현황과 기록
│  └─ references/             참고 캡처 (데이터 점검 체크리스트, 그룹 판정 원칙)
├─ notebook/
│  └─ 01_data_inventory.ipynb 이미지 목록과 분포 그림 생성 (T01)
├─ scripts/
│  ├─ report01_audit.py       무결성·SHA-256·pHash·박스 통계 감사 (REPORT_01)
│  ├─ build_dataset_work.py   CSV 데이터를 툴킷용 작업본으로 변환 (승인자 필수)
│  ├─ make_label_review.py    사람이 채울 상자 검토표 생성
│  ├─ find_duplicates.py      중복 후보, 탐지 성능, 사람이 채울 쌍 검토표 생성
│  ├─ check_data_bundle.py    DATA 묶음 인계 전 점검
│  └─ data_common.py          위 스크립트의 공통 함수
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

Python 3.12 기준이다. Anaconda 환경이면 필요한 패키지가 이미 들어 있다. `python`이 Windows Store 별칭으로 연결되어 실행되지 않으면 Anaconda의 `python.exe` 경로를 직접 쓴다.

```bash
pip install -r requirements.txt

# 1) 이미지 목록 생성 → artifacts/data_inventory.csv  (스크립트보다 먼저)
jupyter nbconvert --to notebook --execute --inplace notebook/01_data_inventory.ipynb

# 2) 1차 감사 → artifacts/report01_*, exact_duplicates.csv 등  (약 1–2분)
python scripts/report01_audit.py
```

### DATA_v001 만들기

팀 합의와 변환 승인 후에 실행한다. 같은 출력 파일이 있으면 덮어쓰지 않고 멈춘다.

```bash
# 1) 작업본 생성 → DATA_v001/dataset_work, review/file_mapping.csv, source_manifest.csv
python scripts/build_dataset_work.py --source . --bundle DATA_v001 --approved-by 승인자

# 2) 기초 검사 → DATA_v001/audit
python guideline/toolkit/audit_dataset.py --root DATA_v001/dataset_work --out DATA_v001/audit --near --overlays 781

# 3) 상자 검토표 → review/label_review.csv, image_meta.csv (판정은 사람이 채운다)
python scripts/make_label_review.py --bundle DATA_v001

# 4) 중복 후보와 쌍 검토표 → review/pair_review.csv 등 (판정은 사람이 채운다, 약 10분)
python scripts/find_duplicates.py --bundle DATA_v001

# 5) 인계 전 점검
python scripts/check_data_bundle.py --bundle DATA_v001 --rerun-audit --open-items DATA_v001/review/open_items.csv
```

학습, 평가 명령은 팀원 2의 분할 인계 후 학습 담당과 팀원 3의 기준에 맞춰 추가한다.

## 데이터 메모

- 주석 형식: CSV(`filename,width,height,class,xmin,ymin,xmax,ymax`), 클래스는 `Microplastic` 하나
- `test` 분할은 없다
- 출처: [Kaggle Microplastic Dataset for Computer Vision](https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision) v1. 합계 용량이 같고 팀원 1이 같은 다운로드 경로임을 확인했다
- Kaggle 표기 라이선스는 CC0이다. 상위 출처의 이용 조건은 미확인이므로 저장소를 공개하기 전에 확인한다
- [논문 저장소](https://github.com/zayedarju/microplastic-detection)의 데이터는 이 데이터와 달라 보인다. 같다고 확인되기 전에는 논문 수치와 비교하지 않는다
- 다운로드 날짜와 원본 압축파일 해시는 아직 기록되지 않았다
