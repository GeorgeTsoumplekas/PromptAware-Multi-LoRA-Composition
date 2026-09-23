#!/usr/bin/env bash
# Generate the W-Switch and W-Composite images used for Tables 1 and 2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/generated}"
if [[ -n "${COMPOSITION_SIZES:-}" ]]; then
  read -r -a SIZES <<< "${COMPOSITION_SIZES}"
else
  SIZES=(2 3 4 5)
fi

mkdir -p "${OUTPUT_DIR}"

run_method() {
  local size="$1"
  shift
  echo
  echo "=== $* --composition_size ${size} ==="
  "${PYTHON_BIN}" "${ROOT}/benchmark_multi_loras.py" \
    --composition_size "${size}" \
    --num_images 3 \
    --output_dir "${OUTPUT_DIR}" \
    --denoise_steps 100 \
    --cfg_scale 7 \
    --lora_scale 0.8 \
    --height 1024 \
    --width 768 \
    --switch_step 5 \
    --lora_path "${ROOT}/models/lora/reality" \
    --lora_info_file "${ROOT}/reality_lora_info.json" \
    "$@"
}

for size in "${SIZES[@]}"; do
  run_method "${size}" \
    --method weighted_switch_adaptive_tailed \
    --ablation_method ablated \
    --reserved_steps 5 \
    --seeds 42 43 44

  run_method "${size}" \
    --method weighted_composite_adaptive \
    --ablation_method triggers \
    --seeds 42 43 44
done

echo
echo "Images written under ${OUTPUT_DIR}"
