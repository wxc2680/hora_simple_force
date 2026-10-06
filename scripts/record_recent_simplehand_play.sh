#!/usr/bin/env bash

set -euo pipefail

GPU_ID=${1:-0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR="$REPO_ROOT/outputs/play_videos"

record() {
    local gpu="$1" repo="$2" run_id="$3" object_type="$4" output="$5" tactile="$6"
    local tactile_args=()
    if [[ "$tactile" == "1" ]]; then
        tactile_args+=(--tactile-panel)
    fi
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    EGL_DEVICE_ID="$gpu" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    python -u "$SCRIPT_DIR/record_simplehand_play.py" \
        --repo "/opt/$repo" \
        --checkpoint "/opt/$repo/outputs/Hand2Hora/$run_id/stage1_nn/best.pth" \
        --output "$OUTPUT_DIR/$output" \
        --object-type "$object_type" \
        --frames 600 \
        --fps 20 \
        --width 960 \
        --height 720 \
        "${tactile_args[@]}" \
        --offline-render \
        --overwrite
}

record "$GPU_ID" hora_simplehand 091301cube cube simplehand_proprio_091301cube_600f.mp4 0
record "$GPU_ID" hora_simplehand_force 091302cube cube simplehand_tactile_091302cube_600f.mp4 1
record "$GPU_ID" hora_simplehand 091401cylinder cylinder simplehand_proprio_091401cylinder_600f.mp4 0
record "$GPU_ID" hora_simplehand_force 091402cylinder cylinder simplehand_tactile_091402cylinder_600f.mp4 1
record "$GPU_ID" hora_simplehand 091401cube_seed43 cube simplehand_proprio_091401cube_seed43_600f.mp4 0
record "$GPU_ID" hora_simplehand_force 091402cube_seed43 cube simplehand_tactile_091402cube_seed43_600f.mp4 1
