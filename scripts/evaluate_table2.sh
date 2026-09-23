#!/usr/bin/env bash
# Table 2: MiniCPM-V 2.6 scores for W-Switch and W-Composite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_DIR="${GENERATED_DIR:-${ROOT}/outputs/generated}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/outputs/llm_evals}"
mkdir -p "${EVAL_DIR}"

runs=(
  "weighted_switch_adaptive_tailed_ablated|42 43 44"
  "weighted_composite_adaptive_triggered|42 43 44"
)

for spec in "${runs[@]}"; do
  method="${spec%%|*}"
  seeds="${spec#*|}"
  for size in 2 3 4 5; do
    image_dir="${GENERATED_DIR}/${method}_${size}"
    if [[ ! -d "${image_dir}" ]]; then
      echo "Missing generated images: ${image_dir}" >&2
      exit 1
    fi
    echo
    echo "=== MiniCPM ${method} N=${size} ==="
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" "${ROOT}/llm.py" \
      --methods "${method}" \
      --paths "${GENERATED_DIR}" \
      --compos_num "${size}" \
      --seeds ${seeds} \
      --image_style reality \
      --lora_path "${ROOT}/models/lora" \
      --lora_info_path lora_info.json \
      --output_txt "${EVAL_DIR}/${method}_${size}.txt"
  done
done

"${PYTHON_BIN}" "${ROOT}/scripts/collect_table2.py" --eval_dir "${EVAL_DIR}"
