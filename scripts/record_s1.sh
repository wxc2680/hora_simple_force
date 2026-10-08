#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?Usage: ./scripts/record_s1.sh GPU_ID SEED RUN_NAME [OBJECT_TYPE]}
SEED=${2:?Usage: ./scripts/record_s1.sh GPU_ID SEED RUN_NAME [OBJECT_TYPE]}
RUN_NAME=${3:?Usage: ./scripts/record_s1.sh GPU_ID SEED RUN_NAME [OBJECT_TYPE]}
OBJECT_TYPE=${4:-cube}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CHECKPOINT="$REPO_ROOT/outputs/Hand2Hora/$RUN_NAME/stage1_nn/best.pth"
OUTPUT="$REPO_ROOT/outputs/play_videos/simplehand_tactile_${RUN_NAME}_best_600f.mp4"

PYOPENGL_PLATFORM=egl \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
EGL_DEVICE_ID="$GPU_ID" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -u "$SCRIPT_DIR/record_simplehand_play.py" \
    --repo "$REPO_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --object-type "$OBJECT_TYPE" \
    --frames 600 \
    --warmup-frames 100 \
    --fps 20 \
    --width 960 \
    --height 720 \
    --seed "$SEED" \
    --tactile-panel \
    --offline-render \
    --overwrite
