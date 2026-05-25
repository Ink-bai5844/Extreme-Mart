#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/data/831}"
EPOCHS="${EPOCHS:-160}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"
PATIENCE="${PATIENCE:-50}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-20}"
CACHE_IMAGES="${CACHE_IMAGES:-0}"
AUTO_RETRY="${AUTO_RETRY:-1}"
CONF_GRID="${CONF_GRID:-0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40}"
NO_CALIBRATE="${NO_CALIBRATE:-0}"
HEAD_OVERSAMPLE="${HEAD_OVERSAMPLE:-2}"
DEVICE="${DEVICE:-}"
GPU_PREFER="${GPU_PREFER:-3090}"
MODEL="${MODEL:-yolov8n.pt}"
PROJECT_DIR="${PROJECT_DIR:-/project/train/runs}"
WORK_DIR="${WORK_DIR:-/project/train/work/helmet_detection_dataset}"
MODEL_OUTPUT="${MODEL_OUTPUT:-/project/train/models/your_model}"
WHEELHOUSE="${WHEELHOUSE:-wheelhouse}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
DEPS_INDEX_URL="${DEPS_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-0}"

cd "$(dirname "$0")"

if [ ! -d "$DATA_DIR" ] && [ -d "/home/data" ]; then
  DATA_DIR="/home/data"
fi

mkdir -p "$(dirname "$MODEL_OUTPUT")"

echo "[start_train] cwd=$(pwd)"
echo "[start_train] DATA_DIR=$DATA_DIR"
echo "[start_train] EPOCHS=$EPOCHS IMGSZ=$IMGSZ BATCH=$BATCH WORKERS=$WORKERS SAVE_PERIOD=$SAVE_PERIOD"
echo "[start_train] PATIENCE=$PATIENCE CLOSE_MOSAIC=$CLOSE_MOSAIC CACHE_IMAGES=$CACHE_IMAGES AUTO_RETRY=$AUTO_RETRY"
echo "[start_train] CONF_GRID=$CONF_GRID NO_CALIBRATE=$NO_CALIBRATE HEAD_OVERSAMPLE=$HEAD_OVERSAMPLE"
echo "[start_train] MODEL=$MODEL"
echo "[start_train] MODEL_OUTPUT=$MODEL_OUTPUT"
echo "[start_train] T4-safe defaults: BATCH=8 WORKERS=2 SAVE_PERIOD=5. Override env vars only if the platform has enough host RAM."

if [ -z "$DEVICE" ] && command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_DEVICE="$(python3 - <<PY
import csv
import subprocess
import sys

prefer = "${GPU_PREFER}".lower()
try:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    sys.exit(0)

gpus = []
for row in csv.reader(output.splitlines()):
    if len(row) < 3:
        continue
    try:
        index = row[0].strip()
        name = row[1].strip()
        memory = int(row[2].strip())
    except ValueError:
        continue
    gpus.append((index, name, memory))

if not gpus:
    sys.exit(0)

preferred = [gpu for gpu in gpus if prefer and prefer in gpu[1].lower()]
chosen = max(preferred or gpus, key=lambda gpu: gpu[2])
print(chosen[0])
PY
)"
  if [ -n "$DETECTED_DEVICE" ]; then
    DEVICE="$DETECTED_DEVICE"
    echo "[start_train] auto selected GPU device=$DEVICE by GPU_PREFER=$GPU_PREFER"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits || true
  fi
fi

if [ -d "$WHEELHOUSE" ] && find "$WHEELHOUSE" -maxdepth 1 -type f \( -name "*.whl" -o -name "*.tar.gz" \) | grep -q .; then
  echo "[start_train] installing requirements from local wheelhouse: $WHEELHOUSE"
  python3 -m pip install --no-index --find-links "$WHEELHOUSE" --no-deps -r requirements-offline.txt
elif [ "$INSTALL_DEPS" = "1" ]; then
  echo "[start_train] installing requirements from $DEPS_INDEX_URL ..."
  python3 -m pip install --no-cache-dir -i "$DEPS_INDEX_URL" -r requirements.txt
else
  python3 - <<'PY'
missing = []
for module in ("ultralytics", "yaml", "PIL", "numpy", "cv2", "torch"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python dependencies: "
        + ", ".join(missing)
        + ". Put Linux wheels under wheelhouse/, preinstall requirements.txt, "
        + "or rerun with INSTALL_DEPS=1 when network access is available."
    )
PY
fi

if [ ! -f "$MODEL" ]; then
  echo "Missing local YOLO weight: $MODEL" >&2
  echo "Put the weight file next to start_train.sh or set MODEL to an existing local .pt path." >&2
  exit 1
fi

DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

CACHE_ARGS=()
if [ "$CACHE_IMAGES" = "1" ]; then
  CACHE_ARGS=(--cache)
fi

CALIBRATE_ARGS=(--conf-grid "$CONF_GRID" --head-oversample "$HEAD_OVERSAMPLE")
if [ "$NO_CALIBRATE" = "1" ]; then
  CALIBRATE_ARGS+=(--no-calibrate)
fi

echo "[start_train] probing dataset..."
python3 -u dataset_probe.py --data "$DATA_DIR" --output dataset_report.json

echo "[start_train] training helmet detection model..."
run_train() {
  local batch="$1"
  local workers="$2"
  local save_period="$3"

  python3 -u train_helmet.py \
    --data "$DATA_DIR" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$batch" \
    --workers "$workers" \
    --save-period "$save_period" \
    --patience "$PATIENCE" \
    --close-mosaic "$CLOSE_MOSAIC" \
    "${CALIBRATE_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    "${CACHE_ARGS[@]}" \
    --workdir "$WORK_DIR" \
    --project "$PROJECT_DIR" \
    --name helmet_detection \
    --model-output "$MODEL_OUTPUT"
}

set +e
run_train "$BATCH" "$WORKERS" "$SAVE_PERIOD"
TRAIN_STATUS=$?
set -e

if [ "$TRAIN_STATUS" -ne 0 ]; then
  echo "[start_train] training failed with exit code $TRAIN_STATUS" >&2
  if [ "$AUTO_RETRY" = "1" ] && { [ "$TRAIN_STATUS" -eq 137 ] || [ "$TRAIN_STATUS" -eq 143 ]; }; then
    RETRY_BATCH=$((BATCH > 1 ? BATCH / 2 : 1))
    RETRY_WORKERS=0
    RETRY_SAVE_PERIOD=-1
    echo "[start_train] process was likely killed by resource limits; retrying once with batch=$RETRY_BATCH workers=$RETRY_WORKERS save_period=$RETRY_SAVE_PERIOD" >&2
    set +e
    run_train "$RETRY_BATCH" "$RETRY_WORKERS" "$RETRY_SAVE_PERIOD"
    TRAIN_STATUS=$?
    set -e
  fi
fi

if [ "$TRAIN_STATUS" -ne 0 ]; then
  echo "[start_train] training failed; no valid model artifacts were produced" >&2
  exit "$TRAIN_STATUS"
fi

echo "[start_train] done"
