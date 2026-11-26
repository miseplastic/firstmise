# feature4 실행·평가 안내

현재 제목은 [RESEARCH_TITLE.txt](../RESEARCH_TITLE.txt), 기준은 [최종 연구 설계](../docs/RESEARCH_PLAN.md)다. 코드 구현과 CPU 합성 검산은 수행했지만, **최종 데이터 검수·서버 GPU pilot·새 12회 본학습·본실험 결과는 완료하지 않았다.**

## 파일과 책임

| 파일 | 하는 일 |
|---|---|
| `exposure/prepare.py` | 검수 완료 CSV·근거를 검사하고 고정 네 조건 입력 생성 |
| `run_training.py`, `exposure/runner.py` | 고정 설정 학습·실패 시도 보존·Val/Test 예측 실행 |
| `exposure/worker.py` | 공식 YOLO 로더·모델·NMS·매칭·AP로 전체 이미지 결과 저장 |
| `exposure/metrics.py`, `exposure/analyze.py` | 계수·FP/FN·임계값·A/B 비교·그룹 부트스트랩 |
| `run_feature4_job.sh` | 할당받은 GPU 작업 안에서 pilot/main 순서 실행 |
| `run_legacy_rg.py`, `build.py` | 과거 R/G용 코드 보존. 새 본실험에 사용하지 않음 |

서승덕은 실행·평가, 박도현은 최종 목록·생성 입력과 코드 검산, 박준현은 그룹·GT·어려운 판정, 양현준은 검수·설계/기록 대조를 맡는다.

## 1. 서버에서 새 브랜치 받기

기존 `firstmise-feature3`와 이전 결과는 그대로 보존한다. 아래 새 폴더가 이미 있으면 해당 폴더에서 `git pull --ff-only`로 갱신한다.

```bash
# Use your existing project parent directory.
git clone --branch feature4 --single-branch https://github.com/miseplastic/firstmise.git firstmise-feature4
cd firstmise-feature4
CONDA_INIT="/path/to/conda.sh"
YOLO="/path/to/yolov5"
source "$CONDA_INIT"
conda activate microplastic-yolov5
python -m unittest discover -s tests -v
```

합성 검산에는 NumPy·Pillow가 필요하다. 학습 환경에는 PyYAML·PyTorch와 기존 YOLO 의존성이 필요하다. 기존 성공 환경을 우선 사용하고 일괄 업그레이드하지 않는다.

확인된 YOLO commit은 `9777cd00e86382dfa5a12e6244ec4f609bc4774c`, 초기 가중치 SHA-256은 `8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee`다. 코드는 이를 검사하고 YOLO 추적 파일 수정 시 중단한다. YOLO 저장소를 임의로 최신화하지 않는다. 수정본이 있다면 변경 내용을 먼저 확인하며 강제 reset하지 않는다.

## 2. 박도현에게 받을 최종 입력

`experiment/.inputs/reviewed_v001/`에 다음 파일을 받는다. 폴더 이름은 예시이며 실제 경로를 명시한다. **현재 저장소에는 승인된 최종 입력이 없다.** 과거 R/G 목록을 이름만 바꿔 사용하지 않는다.

- `manifest.csv`: 전체 원본 풀을 한 행씩 담은 최종 목록.
- `review.json`: CSV 해시와 사람 검토·근거 연결.
- 검수 근거 파일 및 CSV가 가리키는 최종 이미지·YOLO 라벨.

### CSV 계약

템플릿: [manifest.csv](exposure/templates/manifest.csv).

| 열 | 값·규칙 |
|---|---|
| `image_id` | 고유 원본 ID. 예: 출처 폴더/파일명 |
| `group_id` | 최종 검토 그룹. 같은 그룹은 split·A/B를 넘지 않음 |
| `split` | `train`, `val`, `test` |
| `ab` | Test만 `A` 또는 `B`, 나머지는 빈칸 |
| `control_id` | Test마다 독립 Train 원본 하나의 image_id. 중복 사용 금지. 나머지는 빈칸 |
| `image_path`, `label_path` | CSV 위치 기준 상대 경로 또는 실제 절대 경로 |
| `image_sha256`, `label_sha256` | 검수한 실제 파일 SHA-256. 변경 시 재검토 기록·CSV 갱신 |
| `review_status`, `reviewer` | Test·Val·선택 대조는 `reviewed` 및 실제 검토자. 나머지도 검수 범위를 사실대로 기록 |

박도현의 최종 split/A-B·대조표를 이 형식으로 연결한다. 코드가 A/B를 다시 뽑거나 대조를 자동 선정하지 않는다. Test 양쪽과 Val 전체에 AP 계산용 GT가 있어야 한다. 그룹에 GT가 없는 재표집이 나오면 해당 AP 구간은 미보고 상태로 표시한다.

템플릿 [review.json](exposure/templates/review.json)의 `approved`는 기본 false다. 담당자가 실제 완료한 `groups_reviewed`, `gt_reviewed`, `split_reviewed`, `control_reviewed`, `train_known_errors_resolved`에 대해 검토자와 존재하는 근거 파일 경로를 기록한다. AI 코드가 사람 검수를 승인하지 않는다. `manifest_sha256`에는 최종 CSV 파일 해시를 넣는다.

### 고정 입력 만들기·확인

```bash
python -m experiment.exposure.prepare \
  --manifest experiment/.inputs/reviewed_v001/manifest.csv \
  --review experiment/.inputs/reviewed_v001/review.json \
  --out experiment/.inputs/exposure_v001

python experiment/run_training.py --bundle experiment/.inputs/exposure_v001 --check
```

생성기는 대조 원본을 공통 Train에서 제외하고 Test/대조 모두 같은 RGB PNG 반전 처리를 한다. x'=1-x, y·w·h·박스 수 보존. 이미지·라벨·목록·근거·plan을 해시로 고정한다. 실제 노출 이미지 수·학습 장수와 슬롯 매핑을 저장한다. 입력 변경 시 기존 bundle을 덮어쓰지 말고 새 버전을 만든다. YAML·목록의 절대 경로는 실행 시 만든다.

## 3. GPU pilot: 먼저 1에포크 전 과정 확인

서버 로그인 노드에서 모델을 학습하지 않고 기존 `jobsubmit` 방식으로 GPU에 제출한다. 아래 30분은 **pilot의 실행 제한**이며 인수인계 마감 시각이 아니다.

```bash
REPO="$(pwd)"
BUNDLE="$REPO/experiment/.inputs/exposure_v001"
RUN_ROOT="$REPO/experiment/.runtime/exposure_v001"
(
  unset LD_LIBRARY_PATH
  jobsubmit -p 4000ada -c 3 -m 20G -G 1 -t 0-00:30:00 \
    bash "$REPO/experiment/run_feature4_job.sh" pilot "$BUNDLE" "$RUN_ROOT" "$YOLO" "$CONDA_INIT"
)
```

작업 제출 도구가 알려준 로그 위치와 `sacct` 상태를 확인한다. 성공하면 `$RUN_ROOT/pilot/analysis/analysis.json`에 `status=completed`, `is_pilot=true`가 있어야 한다. 이는 0%·seed101·1에포크 예비 실행이며 12회에 포함하지 않는다.

pilot은 학습→last.pt→Val/Test 추론→공식 AP/자체 AP 대조→count matcher 대조→임계값·오류표까지 확인한다. 완료 기록 없이 본학습을 시작하면 실행기가 중단한다. 같은 bundle·가중치·YOLO·코드·worker 설정의 pilot이어야 한다.

## 4. 본학습·Val/Test·분석

pilot 완료와 데이터 준비 근거를 확인한 뒤 실행한다. 아래 4시간 제한은 제출 예시이며 파티션 정책·가용 시간에 맞춰 확인한다. 실제 GPU·장수·설정에 따라 시간을 확인하고 추가 평가·통계·대기도 고려한다.

```bash
REPO="$(pwd)"
BUNDLE="$REPO/experiment/.inputs/exposure_v001"
RUN_ROOT="$REPO/experiment/.runtime/exposure_v001"
(
  unset LD_LIBRARY_PATH
  jobsubmit -p 4000ada -c 3 -m 20G -G 1 -t 0-04:00:00 \
    bash "$REPO/experiment/run_feature4_job.sh" main "$BUNDLE" "$RUN_ROOT" "$YOLO" "$CONDA_INIT"
)
```

기본 실행은 0·50A·50B·100 × seed101·102·103, 입력640·batch4·30에포크·workers2·동일 초기 가중치/모델 cfg·앵커·fliplr=0·noautoanchor·patience=0이다. YOLO에서 patience=0은 early stopping 비활성화다. 실제 opt/hyp와 에포크 완료를 검사하고 `last.pt`만 평가한다.

기본 Job 스크립트는 학습 12회 후 저장·분석한다. 실행기를 직접 사용할 때는 완료한 조건·시드를 `--condition 0 --seed 101`처럼 선택해 먼저 `--predict`할 수 있다. 같은 프로젝트를 동시에 실행하지 않는다.

## 5. 결과 읽기

`$RUN_ROOT/main/analysis/`:

| 파일 | 내용 |
|---|---|
| `threshold_grid.csv` | Val 19개 후보의 MAE·선택값 |
| `scores.csv` | 조건·시드·전체/A/B의 mAP·MAE·FP/FN·Bias·표본 수 |
| `per_image.csv` | 0개 예측 포함 이미지별 GT/예측 수·TP/FP/FN·오차·노출·버전 |
| `contrasts.csv` | 100%-0%, 50A/50B-0%, C_A/C_B, D_A/D_B·평균과 시드별 구간 |
| `seed_summary.csv` | 대응 차이의 세 원값·평균·표본 표준편차 |
| `samples_*.jsonl.gz` | 같은 그룹이 여러 번 뽑힌 횟수까지 보존한 재표집 목록 |
| `analysis.json` | 완료 상태·입력/코드 해시·구간 정의·max_det 도달 이미지 |

두 임계값 방식 `baseline_common`(주 분석)·`model_specific`(민감도)을 함께 계산한다. mAP 원값은 0-1, 차이는 %p. MAE·Bias·FP/FN 평균 차이는 개/이미지다. 개선·악화·상쇄 사례는 이미지별 표와 원본을 함께 읽는다.

각 시드 2,000회, 난수20261124. 전체 주 비교는 전체 그룹 복원추출, A/B 비교는 각 묶음 내 그룹 복원추출. 모델 간 같은 표본 적용·반복 그룹 중복 반영·mAP 재계산. 고정 모델·임계값의 평가 표본 구간이며 시드/임계값 선택/다른 분할 변동을 모두 포함하지 않는다. 그룹 2개 미만 또는 AP가 정의되지 않는 재표집은 해당 구간을 미보고로 표시한다. C 평균과 D 평균은 같은 값이다.

예측 worker는 conf0.001·NMS IoU0.6·max_det300·FP32·TTA끔·GT혼합끔으로 고정한다. official process_batch·ap_per_class를 직접 사용하며 매칭·AP 대조 실패, NMS 시간 초과, 이미지 누락이면 export 완료를 만들지 않는다. 완료된 0개 예측은 JSONL에 빈 predictions로 남긴다. 최대 검출 상한 도달은 analysis.json에서 확인한다.

## 6. 중단·재실행

- 실행 완료는 재사용하되 입력·가중치·코드 해시가 달라지면 거절한다. 변경된 실험은 새 bundle·project 버전으로 시작한다.
- 실패한 학습·예측 로그를 확인한 후 동일 명령에 `--retry-failed`를 추가한다. 새 attempt에 저장해 이전 기록을 보존한다. 종료된 성공 시도를 새 시드로 교체하지 않는다.
- Job이 강제 종료되어 `.pipeline.lock`이 남으면 실제 해당 프로젝트 프로세스가 전부 종료됐는지 먼저 확인한다. 그 후 그 프로젝트의 lock만 제거해 재시도한다. 실행 중 lock을 지우지 않는다.
- 분석 출력 폴더는 덮어쓰지 않는다. 계산을 재실행할 때 `--out`에 새 폴더를 사용한다. 본실험 분석은 `python -m experiment.exposure.analyze --bundle ... --project ... --out ...`로 별도 실행 가능하다.
- 기존 R/G 재현이 필요하면 보존된 `run_legacy_rg.py`를 쓰며 새 본실험과 결과를 섞지 않는다.

## 검증 범위

합성 데이터로 그룹·A/B·대조·해시·반전·지표·임계값·반복 그룹·세 시드 집계·pilot/main 분리를 검산했다. 학습 호출은 로컬 테스트에서 모의 실행했다. 이 PC에는 서버 GPU/PyTorch 환경이 없으므로 실제 YOLO 로딩·학습·추론은 **서버 pilot에서 아직 검증해야 한다.** 최종 데이터의 정답 판정도 이 코드 구현으로 완료되지 않는다.
