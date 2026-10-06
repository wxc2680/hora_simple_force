#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TACSL_PYTHON=${TACSL_ISAACGYM_PYTHON:-/opt/tacsl/IsaacGym_Preview_TacSL_Package/isaacgym/python}
TACSL_BINDING="$TACSL_PYTHON/isaacgym/_bindings/linux-x86_64"

if [[ ! -f "$TACSL_BINDING/gym_38.so" ]]; then
    echo "TacSL Isaac Gym binding not found: $TACSL_BINDING/gym_38.so" >&2
    exit 1
fi

export PYTHONPATH="$TACSL_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$TACSL_BINDING:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
if [[ $# -eq 0 ]]; then
    set -- python scripts/smoke_simplehand_vt_refine.py --num-envs 4 --steps 8
fi
exec "$@"
