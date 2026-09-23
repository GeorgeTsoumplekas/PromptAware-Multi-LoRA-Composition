#!/usr/bin/env bash
# Crop character faces from generated images with S3FD + FAN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_DIR="${GENERATED_DIR:-${ROOT}/outputs/generated}"
CROPPED_DIR="${CROPPED_DIR:-${ROOT}/outputs/cropped}"

shopt -s nullglob
dirs=("${GENERATED_DIR}"/*/)
if [[ ${#dirs[@]} -eq 0 ]]; then
  echo "No generated folders found in ${GENERATED_DIR}" >&2
  exit 1
fi

for dir in "${dirs[@]}"; do
  name="$(basename "${dir}")"
  output_dir="${CROPPED_DIR}/${name}_cropped"
  mkdir -p "${output_dir}"
  echo "Cropping faces: ${dir} -> ${output_dir}"
  "${PYTHON_BIN}" "${ROOT}/face_cropping/crop_faces.py" \
    --image_dir "${dir%/}" \
    --output_dir "${output_dir}"
done
