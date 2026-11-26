#!/usr/bin/env bash
# Execute inside an allocated GPU job. Supply machine-specific paths as arguments.
set -euo pipefail
MODE="${1:?Usage: bash experiment/run_feature4_job.sh pilot|main BUNDLE RUN_ROOT YOLO CONDA_INIT [ENV_NAME]}"
BUNDLE="${2:?Provide the prepared reviewed bundle directory}"
RUN_ROOT="${3:?Provide a versioned output root directory}"
YOLO="${4:?Provide the verified YOLO checkout directory}"
CONDA_INIT="${5:?Provide the local conda.sh path}"
ENV_NAME="${6:-microplastic-yolov5}"
if [[ "$MODE" != pilot && "$MODE" != main ]]; then
  echo "Mode must be pilot or main" >&2; exit 2
fi
set +u
source "$CONDA_INIT"
conda activate "$ENV_NAME"
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$YOLO/yolov5s.pt"
cd "$REPO"
python -c 'import torch; assert torch.cuda.is_available(), "No allocated CUDA GPU"; print(torch.cuda.get_device_name(0))'
python experiment/run_training.py --bundle "$BUNDLE" --check
PROJECT="$RUN_ROOT/$MODE"
if [[ "$MODE" == pilot ]]; then
  python experiment/run_training.py --bundle "$BUNDLE" --yolov5 "$YOLO" --weights "$WEIGHTS" \
    --project "$PROJECT" --pilot
else
  python experiment/run_training.py --bundle "$BUNDLE" --yolov5 "$YOLO" --weights "$WEIGHTS" \
    --project "$PROJECT" --pilot-report "$RUN_ROOT/pilot/analysis/analysis.json"
fi
python experiment/run_training.py --bundle "$BUNDLE" --yolov5 "$YOLO" --project "$PROJECT" --predict
if [[ "$MODE" == pilot ]]; then
  python -m experiment.exposure.analyze --bundle "$BUNDLE" --project "$PROJECT" --out "$PROJECT/analysis" --pilot
else
  python -m experiment.exposure.analyze --bundle "$BUNDLE" --project "$PROJECT" --out "$PROJECT/analysis"
fi
