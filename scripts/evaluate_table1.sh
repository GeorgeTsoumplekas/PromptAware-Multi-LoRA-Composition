#!/usr/bin/env bash
# Table 1 metrics for W-Switch and W-Composite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_DIR="${GENERATED_DIR:-${ROOT}/outputs/generated}"
CROPPED_DIR="${CROPPED_DIR:-${ROOT}/outputs/cropped}"
REAL_ROOT="${REAL_ROOT:-${ROOT}/concept_images}"
CROPPED_REAL_ROOT="${CROPPED_REAL_ROOT:-${ROOT}/concept_images_cropped}"

METHODS=(
  weighted_switch_adaptive_tailed_ablated
  weighted_composite_adaptive_triggered
)
SIZES=(2 3 4 5)
NON_FACE_CONCEPTS=(
  clothing_1 clothing_2
  background_1 background_2
  object_1 object_2
  style_1 style_2
)

for method in "${METHODS[@]}"; do
  for size in "${SIZES[@]}"; do
    generated="${GENERATED_DIR}/${method}_${size}"
    cropped="${CROPPED_DIR}/${method}_${size}_cropped"
    echo
    echo "=== ${method} N=${size} ==="

    if [[ ! -d "${generated}" ]]; then
      echo "Missing generated images: ${generated}" >&2
      exit 1
    fi
    if [[ ! -d "${cropped}" ]]; then
      echo "Missing cropped images: ${cropped}" >&2
      echo "Run scripts/crop_faces.sh and place the SAM3 crops first." >&2
      exit 1
    fi
    for concept in "${NON_FACE_CONCEPTS[@]}"; do
      if [[ ! -d "${cropped}/${concept}" ]]; then
        echo "Missing SAM3 crop folder: ${cropped}/${concept}" >&2
        exit 1
      fi
    done

    "${PYTHON_BIN}" "${ROOT}/evaluation_scripts/calculate_text_alignment_clip.py" \
      --generated_image_folder "${generated}" \
      --lora_info_file "${ROOT}/reality_lora_info.json" \
      --lora_path "${ROOT}/models/lora/reality"

    "${PYTHON_BIN}" "${ROOT}/evaluation_scripts/calculate_image_alignment_clip.py" \
      --generated_root "${cropped}" \
      --style_reference_root "${REAL_ROOT}" \
      --cropped_reference_root "${CROPPED_REAL_ROOT}" \
      --similarity_method max

    "${PYTHON_BIN}" "${ROOT}/evaluation_scripts/calculate_image_alignment_dino.py" \
      --generated_root "${cropped}" \
      --style_reference_root "${REAL_ROOT}" \
      --cropped_reference_root "${CROPPED_REAL_ROOT}" \
      --similarity_method max

    "${PYTHON_BIN}" "${ROOT}/evaluation_scripts/calculate_identity_alignment_arcface.py" \
      --real_image_folder "${CROPPED_REAL_ROOT}" \
      --generated_image_folder "${cropped}" \
      --similarity_method max
  done
done

"${PYTHON_BIN}" "${ROOT}/scripts/collect_table1.py" \
  --generated_dir "${GENERATED_DIR}" \
  --cropped_dir "${CROPPED_DIR}"
