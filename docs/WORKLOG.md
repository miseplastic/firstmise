# 작업 현황 및 기록

## 현재 목표

전체 학습 전에 데이터 구조와 중복 감사의 실행 가능성을 확인한다. 모델 점수는 이 확인 다음의 목표다.

## 작업 목록

| ID | 작업 | 담당 | 상태 | 완료 기준 | 산출물 |
|---|---|---|---|---|---|
| T01 | 데이터 구조 검증 | 미정 | 완료 | 분할별 이미지 수, 주석 행 수, 클래스, 파일명 토큰, 디스크와 CSV의 불일치를 담은 목록을 저장한다 | artifacts/data_inventory.csv |
| T02 | 완전 중복 감사 | 미정 | 대기 | train 내부, valid 내부, train-valid 사이의 SHA-256 일치 쌍과 파일 목록을 저장한다 | exact_duplicates.csv |
| T03 | 유사 이미지 표본 검토 | 미정 | 대기 | pHash 후보와 사람 판정, 판정 근거를 저장한다. 회전·반전 비교는 표본을 본 뒤에 정한다 | similarity_review.csv |
| T04 | 그룹 및 분할 생성 | 미정 | 대기 | 확정된 그룹이 서로 다른 분할에 걸치지 않는지 검증한 목록을 저장한다. 그룹 확정은 T03 이후다 | split_manifest.csv |
| T05 | 학습 파이프라인 점검 | 미정 | 대기 | 확정 전 설정으로 짧은 학습과 평가가 끝까지 실행된 로그를 남긴다. 이 점수는 비교 결과에 넣지 않는다 | 실행 로그 |
| T06 | 본 실험 | 미정 | 대기 | RESEARCH_PLAN.md의 미확정 항목이 결정 기록에 채워진 뒤, 확정 조건별 3회 반복 지표를 저장한다 | metrics.csv |

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
