#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${COOP2_CONDA_BIN:-$HOME/miniconda3/condabin/conda}"
CONDA_ENV="${COOP2_CONDA_ENV:-behavior51}"
OUTPUT_ROOT="${COOP2_OUTPUT_ROOT:-$ROOT/log}"

export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH="$ROOT/datasets"

cd "$ROOT"

for set_size in 1 3 5; do
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
        python -u -m coop2.experiment.run_s1_grid \
        --modes individual \
        --tasks v4_s1_v4_ll v4_s1_v4_lh v4_s1_v4_hl v4_s1_v4_hh \
        --layout "coop2/team_layouts/s1/sets_${set_size}.json" \
        --scene Merom_1_int \
        --room living_room_0 \
        --steps 12000 \
        --seed 0 \
        --time-limit-seconds 0 \
        --model gpt-5.6-luna \
        --output-root "$OUTPUT_ROOT" \
        --run-timeout-minutes 120 \
        "$@"
done
