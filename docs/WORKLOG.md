# 작업 현황 및 기록

## 현재 목표

팀원 2가 그룹을 확정할 수 있도록 `DATA_v001`을 인계한다. 단계별 계획은 [ROADMAP.md](ROADMAP.md), 팀 합의 안건은 [TEAM_AGENDA.md](TEAM_AGENDA.md)에 있다. 모델 점수는 이 저장소의 당면 목표가 아니다.

## 작업 목록

| ID | 작업 | 담당 | 상태 | 완료 기준 | 산출물 |
|---|---|---|---|---|---|
| T01 | 데이터 구조 검증 | 팀원 1 | 완료 | 분할별 이미지 수, 주석 행 수, 클래스, 파일명 토큰, 디스크와 CSV의 불일치를 담은 목록을 저장한다 | artifacts/data_inventory.csv |
| T02 | 완전 중복 감사 | 팀원 1 | 파일 바이트 기준 완료, 픽셀 기준 대기 | train 내부, valid 내부, train-valid 사이의 SHA-256 일치 쌍과 파일 목록을 저장한다 | artifacts/exact_duplicates.csv, DATA_v001/audit/exact_duplicates.csv |
| T03 | 유사 이미지 1차 판정 | 팀원 1 | 대기 | 후보와 사람 판정, 판정 근거, 검토·미검토 수를 저장한다 | DATA_v001/review/pair_review.csv |
| T04 | 그룹 및 분할 생성 | 팀원 2 | 대기 | 확정된 그룹이 서로 다른 분할에 걸치지 않는지 검증한 목록을 저장한다. 그룹 확정은 T03 이후다 | groups_reviewed.csv, manifest.csv |
| T05 | 학습 파이프라인 점검 | 학습 담당 | 대기 | 확정 전 설정으로 짧은 학습과 평가가 끝까지 실행된 로그를 남긴다. 이 점수는 비교 결과에 넣지 않는다 | 실행 로그 |
| T06 | 본 실험과 평가 | 학습 담당, 팀원 3 | 대기 | RESEARCH_PLAN.md의 미확정 항목이 결정 기록에 채워진 뒤, 확정 조건별 지표를 저장한다 | metrics.json, summary.csv |

이미지별 해시, 그룹, 분할과 실험별 수치는 CSV에 저장한다. 이 파일과 RESEARCH_PLAN.md에는 그 CSV의 위치, 기준, 해석을 적는다.

## 작업 기록

### 2026-10-27

- 수행한 작업: `train`, `valid`의 jpg 수와 `_annotations.csv`를 대조했다. 파일명에서 접두와 번호 토큰을 세어 train/valid 공유 여부를 집계했다. 연구 기준과 작업 기록을 README.md, RESEARCH_PLAN.md, 이 파일로 나눴다.
- 확인된 사실: 로컬 이미지는 train 577장, valid 204장, 합계 781장이다. 주석은 CSV이고 클래스명은 `Microplastic` 하나다. `test` 디렉터리는 없다. 디스크와 CSV의 파일명 불일치는 0건이다. 번호 토큰 기준으로 valid 149개 중 147개가 train에도 있다. 이 겹침은 파일명 관찰이며 동일 원본 판정은 아니다. 발표자료의 783 files와는 2장 차이가 있고 원인은 모른다. 상세 수치는 RESEARCH_PLAN.md 1절에 있다.
- 생성한 파일: `README.md`, `RESEARCH_PLAN.md`, `WORKLOG.md`. `data_inventory.csv`는 만들지 않았다.
- 문제 또는 미확정 사항: 출처, 버전, 다운로드 날짜가 없다. SHA-256과 pHash는 실행하지 않았다. 손상 이미지 여부는 열어서 확인하지 않았다. test 구성, 분할 seed, 학습 seed, 3회 반복 때 어떤 seed를 바꿀지, 학습 스텝, YOLOv5 버전은 미확정이다.
- 다음 작업: T01을 끝내 `data_inventory.csv`를 저장하고, 출처와 783장과의 차이를 적는다. 이어서 T02의 SHA-256 일치 목록을 만든다.
- 연구 계획 변경 여부: 분할 비율, pHash 임계값, 에폭, 배치를 기준값으로 적지 않는다. 해당 항목은 RESEARCH_PLAN.md 4절에 미확정으로 둔다.

### 2026-10-27

- 수행한 작업: Kaggle 노트북 `notebook/notebookf7f9c1f971.ipynb`의 주석 로드와 분포 그림을 로컬 경로용 `notebook/01_data_inventory.ipynb`로 옮기고 실행했다. Gemini, MiMo-VL-7B, LoRA 학습 셀은 실행하지 않았다.
- 확인된 사실: 저장한 목록은 train 577장, valid 204장, 박스 5,425 / 1,701, 클래스 `Microplastic`, 파일 누락 0건이다. 번호 토큰 공유는 147개로 1절과 같다.
- 생성한 파일: `artifacts/data_inventory.csv`, `artifacts/dataset_overview.png`, `artifacts/sample_boxes.png`.
- 문제 또는 미확정 사항: 출처, 다운로드 날짜, 공개 페이지 783장과의 차이는 그대로다. 파일명 번호가 같은 것은 아직 중복 판정이 아니다.
- 다음 작업: T02. 목록의 이미지 경로로 train 내부, valid 내부, train-valid 사이의 SHA-256 일치 쌍을 저장한다.
- 연구 계획 변경 여부: 없다.

### 2026-11-03 (2026-11-06에 REPORT_01.md를 근거로 사후 기록)

- 수행한 작업: `scripts/report01_audit.py`로 무결성, 파일 바이트 SHA-256, pHash, 박스 통계, 박스 배치를 분석하고 `docs/REPORT_01.md`를 작성했다.
- 확인된 사실: 781장 모두 디코딩되고 크기가 일치한다. SHA-256 일치는 0쌍이다. 전역 pHash는 거리 0인 쌍도 다른 장면이어서 판별력이 낮다. valid의 66%가 `c` 접두다.
- 생성한 파일: `docs/REPORT_01.md`, `artifacts/`의 `report01_*`, `exact_duplicates.csv`, `phash_*`, `box_layout_candidates.csv`.
- 문제 또는 미확정 사항: 라벨 내용 검토와 사람 검토는 하지 않았다.
- 연구 계획 변경 여부: 없다. 보고서 7절에 제안만 적었다.

### 2026-11-06

- 수행한 작업: 팀원 1·2 매뉴얼과 툴킷을 대조해 로드맵을 만들었다. `original/`과 기존 데이터를 대조했다. Kaggle 메타데이터와 논문 저장소를 확인했다. 임시 환경에서 툴킷 연습 흐름을 실행했다. 문서의 표현, 상태, 미확정 항목을 정리했다.
- 확인된 사실: `original/`과 `train/`, `valid/`는 파일 783개의 SHA-256이 모두 같다. 합계 용량은 Kaggle v1과 같다. 두 매뉴얼의 툴킷과 템플릿은 파일이 모두 같다. 합성 데이터의 연습 흐름은 Python 3.12에서 통과했다. 상세 내용은 REPORT_01.md의 "2026-11-06 추가 확인"에 있다.
- 생성·수정한 파일: `docs/ROADMAP.md`, `docs/TEAM_AGENDA.md` 생성. `README.md`, `docs/RESEARCH_PLAN.md`, `docs/REPORT_01.md`, 이 파일 수정.
- 문제 또는 미확정 사항: 실제 데이터로 툴킷을 실행하지 않았다. 이번 수치는 임시 스크립트의 결과이며 저장소에 재현 스크립트가 없다. 원 논문은 접근이 차단되어 읽지 못했다. 10월 27일 기록의 `notebook/notebookf7f9c1f971.ipynb`는 저장소에 없다. `C:\miseplastic`에 같은 구조의 사본이 있어 정본 여부를 확인해야 한다.
- 다음 작업: TEAM_AGENDA.md를 팀에 전달하고 답을 기록한다. 이어서 ROADMAP.md 1단계를 진행한다.
- 연구 계획 변경 여부: RESEARCH_PLAN.md 8절에 범위와 판정 값 결정을 추가했다. 나머지 기본안은 제안으로만 적었다.

### 2026-11-06 (1단계 연습)

연습 자료의 결과이며 연구 결과가 아니다.

- 수행한 작업: `guideline/toolkit`에서 `make_demo.py`, `audit_dataset.py`를 `--near` 없이 한 번, 포함해 한 번 실행했다. 오버레이 20장과 중복·후보 쌍을 AI가 열어 판정 예시를 작성했다.
- 확인된 사실: 이미지 30장, 객체 30개, 오류 0개, 완전 중복 15그룹으로 매뉴얼 기대값과 같다. `--near`를 켜면 후보가 17쌍 나온다. 매뉴얼은 후보가 비어 있어도 정상이라고 적었는데, 이는 `--near` 없이 실행한 경우에 맞다. 17쌍은 모두 다른 원본이다.
- 생성한 파일: `guideline/toolkit/`의 `practice/`, `audit_demo/`, `audit_demo_near/`, `practice_review/` (모두 git 제외).
- 문제 또는 미확정 사항: 매뉴얼의 Python 3.11 대신 Anaconda Python 3.12와 Pillow 10.4를 사용했다. 사람이 직접 상자를 보고 판정하는 연습은 아직 남아 있다.
- 다음 작업: 사람이 연습 판정을 확인한다. 2단계 이후는 팀 답변을 받은 뒤 진행한다.
- 연구 계획 변경 여부: 없다.

### 2026-11-06 (인계 스크립트 작성)

- 수행한 작업: `scripts/`에 `build_dataset_work.py`, `make_label_review.py`, `find_duplicates.py`, `check_data_bundle.py`, `data_common.py`를 작성했다. 임시 폴더에서 실제 이미지 사본으로 전체 흐름을 시험했다. 점검 스크립트는 가짜 판정을 넣은 임시 사본으로 시험하고 사본을 지웠다.
- 확인된 사실: 임시 변환에서 이미지 781장, 박스 7,126개의 왕복 불일치는 0건이다. 임시 audit에서 ERROR 0건, 픽셀 기준 완전 중복 0그룹, dHash 후보 1,256쌍이다. `make_official.py`는 test가 없어 예상대로 멈췄다.- 생성한 파일: 위 스크립트. 실제 `DATA_v001`은 만들지 않았다.
- 문제 또는 미확정 사항: 시험 결과는 임시 산출물이며 인계 자료가 아니다. 쌍 판정은 하지 않았다. 검토 범위와 표본 수는 제안값이다.
- 다음 작업: 팀 답변과 변환 승인을 받은 뒤 README.md의 순서대로 `DATA_v001`을 만든다.
- 연구 계획 변경 여부: 없다.
