#!/usr/bin/env bash

set -euo pipefail

GPU_ID=${1:-1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR="$REPO_ROOT/outputs/play_videos"

PYOPENGL_PLATFORM=egl \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
EGL_DEVICE_ID="$GPU_ID" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -u "$SCRIPT_DIR/record_simplehand_play.py" \
    --repo /opt/hora_simplehand \
    --checkpoint /opt/hora_simplehand/outputs/Hand2Hora/081301/stage1_nn/best.pth \
    --output "$OUTPUT_DIR/simplehand_proprio_081301_600f.mp4" \
    --frames 600 \
    --fps 20 \
    --width 960 \
    --height 720 \
    --offline-render \
    --overwrite

PYOPENGL_PLATFORM=egl \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
EGL_DEVICE_ID="$GPU_ID" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -u "$SCRIPT_DIR/record_simplehand_play.py" \
    --repo /opt/hora_simplehand_force \
    --checkpoint /opt/hora_simplehand_force/outputs/Hand2Hora/081302/stage1_nn/best.pth \
    --output "$OUTPUT_DIR/simplehand_tactile_081302_600f.mp4" \
    --frames 600 \
    --fps 20 \
    --width 960 \
    --height 720 \
    --tactile-panel \
    --offline-render \
    --overwrite
