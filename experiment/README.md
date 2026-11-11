# Feature3 학습 준비 기록과 Q&A

작성일 2026-11-11 · 입력 기준 [feature2 커밋 1499cc5](https://github.com/miseplastic/firstmise/commit/1499cc59ead75c8c0cbcc16ce4850b275e146c34)

## 쉬운 설명

사진은 781장이고, 각 사진 속 미세플라스틱 위치를 표시한 박스는 총 7,126개입니다. 사진과 정답 박스를 YOLOv5가 읽는 형식으로 준비했습니다. 애매했던 8장은 입자 배치가 달라 각각 한 장짜리 그룹으로 넣었습니다. 또 `d--12`가 이름이 비슷한 `d--1` 쪽에 들어가던 오류를 고쳐 `d--30`과 묶었습니다.

R은 사진을 한 장씩 섞습니다. G는 같은 그룹으로 정한 사진을 한 묶음으로 옮깁니다. 두 방식 모두 학습·검증·시험 세트를 만들고, 섞는 난수를 바꿔 세 번씩 반복합니다. 같은 반복에서는 R과 G의 학습 사진 수와 모델 학습 시드를 맞춥니다.

## 학습 서버에서 실행

기존에 시험 학습에 성공한 `microplastic-yolov5` 환경과 `~/workspace/microplastic/yolov5/yolov5s.pt`를 사용하는 예시입니다. 학교 서버에서는 GPU가 할당된 작업에서 학습 명령을 실행하세요. 경로가 다르면 `YOLO`와 `WEIGHTS`만 바꾸면 됩니다.

ZIP으로 받은 경우에는 서버에서 압축을 풀고 첫 번째 `git clone` 줄을 건너뛰세요.

```bash
git clone --branch feature3 --single-branch https://github.com/miseplastic/firstmise.git firstmise-feature3
cd firstmise-feature3
source /mnt/test-k8s/anaconda3/etc/profile.d/conda.sh
conda activate microplastic-yolov5
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

YOLO="$HOME/workspace/microplastic/yolov5"
WEIGHTS="$YOLO/yolov5s.pt"
python experiment/run_training.py --check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
git -C "$YOLO" rev-parse HEAD
sha256sum "$WEIGHTS"
```

30 epoch은 기존 설정 제안의 출발값이고, batch 4는 기존 GPU 실습에서 사용한 값입니다. 팀에서 확정한 공통값이 있으면 한 번에 바꿔 여섯 실행에 동일하게 적용하세요. 아래 명령은 각 실행을 동일한 초기 가중치에서 시작합니다. 첫 실행만 하려면 `--run R_1`만 남기세요.

```bash
python experiment/run_training.py \
  --run R_1 --run G_1 --run R_2 --run G_2 --run R_3 --run G_3 \
  --yolov5 "$YOLO" --weights "$WEIGHTS" \
  --epochs 30 --batch-size 4 --imgsz 640 --workers 2 --device 0
```

실행별 `opt.yaml`, `hyp.yaml`, `results.csv`, `weights/best.pt`는 `experiment/.runtime/runs/<실행명>/`에 저장됩니다. 실행용 절대 경로가 적힌 `data.yaml`은 `experiment/.runtime/runs/configs/<실행명>/`에 저장됩니다. 기존 실행 폴더가 있다면 새 `--project` 경로를 지정하고, 이어서 학습할 때는 해당 실행의 `last.pt`와 YOLOv5의 `--resume`을 사용합니다.

PyTorch 2.6 이상에서 순정 v7.0이 공식 체크포인트를 열 때 `weights_only` 오류가 발생하면, 기존에 성공한 서버 환경과 YOLO 코드를 우선 사용하세요. 순정 v7.0과 출처가 확인된 공식 `yolov5s.pt`를 사용하는 경우에는 위 실행 명령 앞에 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`을 붙일 수 있습니다. [YOLOv5 v7.0 학습 코드](https://github.com/ultralytics/yolov5/blob/v7.0/train.py) · [공식 v7.0 가중치](https://github.com/ultralytics/yolov5/releases/tag/v7.0)

## 작업 기록

1. `feature2`를 기준으로 `feature3`를 만들었습니다. 원본 `train/` 577장과 `valid/` 204장 및 각 CSV를 읽어 `experiment/data/dataset_work/`에 YOLO 학습용 이미지와 라벨을 구성했습니다. `valid`의 작업본 경로는 `val`입니다. 이 폴더명은 출처를 나타내며, 실제 학습 배정은 각 실행의 목록을 따릅니다.
2. 클래스는 CSV의 `Microplastic` 하나로 유지했고 ID는 `0`입니다. CSV의 `xmin, ymin, xmax, ymax`를 픽셀 모서리 좌표로 해석해 정규화된 중심 좌표와 너비·높이로 변환했습니다. 모든 좌표를 픽셀로 되돌려 원본과 비교한 결과 불일치 0건, 복사 이미지 해시 불일치 0건입니다. 변환 좌표 관례를 출처에서 별도로 확인한 기록은 없습니다. [변환 결과](data/review/conversion_report.json) · [원본 해시](data/review/source_manifest.csv) · [파일 대응표](data/review/file_mapping.csv)
3. `1`, `110`, `a--84`, `b--122`, `b--84`, `c--1`, `f--200`, `f--94`를 각각 고유한 `solo_*` 그룹으로 바꿨습니다. `d--12`는 `G_D_1_28`에서 `G_D_12_30`으로 수정했습니다. `d--1` 문자열에 `d--12`가 포함되는 문제를 피하기 위해 정확한 접두와 번호로 매핑합니다. 네 D 쌍은 feature2 판정 문서의 쌍을 적용했습니다. [9건 변경표](group_changes.csv) · [전체 그룹](groups.csv)
4. 누락됐던 공용 모듈을 분할 코드와 함께 배치했습니다. 그룹 CSV에는 감사 도구의 `train/파일명` 또는 `val/파일명`을 사용하며, 781개 ID가 정확히 한 번씩 맞아야 생성됩니다. 분할 코드는 미해결 보류 ID를 발견하면 중단합니다.
5. R은 정렬된 이미지 목록을, G는 정렬된 그룹 목록을 각 시드로 섞습니다. 약 70/15/15로 나누고 같은 반복의 학습 장수를 맞춥니다. `R_1`의 `f--108`과 `R_3`의 `d--7`이 장수 맞춤 과정에서 각각 해당 반복의 학습 목록에서 빠졌습니다. [제외 로그](splits/excluded_train.csv)
6. 이미지 목록에는 목록 파일 기준 상대 경로를 기록했습니다. YOLOv5 v7.0은 YAML의 상대 경로를 YOLOv5 저장소 기준으로 해석하므로, 실행기가 로컬 절대 경로를 가진 `data.yaml`을 만듭니다. 저장소를 다른 위치나 운영체제로 옮긴 뒤에도 같은 목록으로 실행할 수 있게 구성했습니다. [YOLOv5 설정 처리](https://github.com/ultralytics/yolov5/blob/v7.0/utils/general.py) · [YOLOv5 목록 처리](https://github.com/ultralytics/yolov5/blob/v7.0/utils/dataloaders.py)

## 분할 결과와 검증

| 반복 | 분할 시드 | 학습 시드 | R 학습 / 검증 / 시험 | G 학습 / 검증 / 시험 |
|---|---:|---:|---:|---:|
| 1 | 11 | 101 | 546 / 117 / 117 | 546 / 117 / 118 |
| 2 | 22 | 102 | 547 / 117 / 117 | 547 / 117 / 117 |
| 3 | 33 | 103 | 546 / 117 / 117 | 546 / 117 / 118 |

- 파일 감사에서 읽을 수 있는 이미지 781장, 박스 7,126개, 형식 오류 0건, 픽셀 완전 중복 0그룹을 확인했습니다. 이번 빌드에서는 유사 해시 후보 탐색을 다시 실행하지 않았습니다. [감사 요약](audit_summary.json)
- 18개 목록의 이미지·라벨 경로와 분할표의 ID가 일치했습니다. 한 실행 안에서 동일 이미지가 두 세트에 들어간 경우는 0건이고, G에서 그룹이 세트를 넘은 경우도 0건입니다. R에서는 매 반복 학습과 시험 사이에 걸치는 D 그룹이 1~2개 있었습니다. [분할표](splits/manifest.csv)
- 모든 세트에 `Microplastic` 라벨이 있습니다. 세트별 박스 면적 중앙값은 이미지 면적의 약 0.246~0.320%였습니다. [분포표](splits/distribution.csv) · [박스 크기](splits/box_sizes.csv)
- 공식 YOLOv5 v7.0(`915bbf2`)의 CPU 데이터 로더로 18개 목록을 모두 읽었습니다. 각 목록의 이미지·박스 수가 분포표와 일치했고, 각각 640×640 이미지 4장으로 된 첫 배치를 읽었습니다. 학교 GPU에서의 모델 학습은 별도입니다.
- 별도 경로에서 재빌드한 결과 생성 파일 1,599개가 바이트 단위로 일치했습니다. `manifest.csv` SHA-256은 `ffea5adc5475490a55d956bd96ab93289439a8ebbb9813cc9d0fca5d6e419d5a`입니다.

## Q&A

**Q. pHash가 0인데 왜 같은 그룹으로 두지 않았나요?**
`c--1`과 `110`은 원형 용기 테두리가 비슷하고, 입자의 위치와 라벨 수는 다릅니다. 회전과 반전을 적용한 박스 배치 점수도 최대 0.054였습니다. 보류 목록의 나머지 두 묶음도 서로 다른 입자 배치를 보였습니다. 시료 ID가 없어 촬영에 쓰인 실제 시료의 동일 여부는 확인하지 못했습니다.

**Q. 네 D 쌍은 모두 완전히 같은 사진인가요?**
픽셀 완전 중복은 0그룹입니다. D 쌍은 feature2 문서가 지정한 비교 실험용 그룹입니다. 일부 쌍은 입자 위치나 모양이 달라 단순 반전·회전 설명을 다시 확인할 여지가 있습니다. 이번 빌드에서는 문서의 쌍을 유지하고 `d--12`의 코드 매핑을 그 문서에 맞췄습니다.

**Q. 왜 어떤 실행은 합계가 780장인가요?**
같은 반복의 R과 G에 같은 수의 학습 이미지를 주기 위해 `R_1`과 `R_3` 학습에서 각 한 장만 뺐습니다. 원본 데이터에는 781장이 모두 있습니다. G는 그룹 수를 기준으로 나누므로 시험 이미지 수가 117장 또는 118장이 됩니다.

**Q. R과 G의 점수 차이를 어떻게 설명하나요?**
R에는 같은 그룹 사진이 학습과 시험에 나뉘는 경우가 있고 G에는 없습니다. 또한 조건별 시험 목록과 박스 분포가 다릅니다. 점수 차이에는 그룹 노출 가능성과 시험 세트 구성의 차이가 함께 반영됩니다. 모델 선택은 검증 세트로 하고, 최종 평가는 각 실행의 시험 세트로 합니다.

**Q. 형식 검증이 끝났으면 모든 라벨이 정확한 건가요?**
이번 검증 범위는 원본과 작업본 이미지의 일치, 박스 변환, 경로, 그룹 배치, 분할 목록입니다. 입자 누락이나 박스 경계의 의미상 정확성은 별도의 육안 검수 항목입니다.

**Q. 분할을 다시 만들려면 어떻게 하나요?**
저장소 루트에서 Pillow를 설치하고 새 경로를 지정합니다. 결과 비교에는 `groups.csv`, `splits/manifest.csv`, 실행별 목록을 사용합니다.

```bash
python -m pip install "Pillow>=10.3"
python experiment/build.py --out ../feature3-rebuild
```
