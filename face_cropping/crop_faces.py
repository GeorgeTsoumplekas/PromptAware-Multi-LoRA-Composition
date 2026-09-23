import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_detection import LandmarksEstimation, preprocess_image

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHARACTER_SUBFOLDERS = ("character_1", "character_2", "character_3")
NAME_PATTERN = re.compile(
    r"^character_(?P<char>[123])_.*_(?P<x>[^_]+)_seed(?P<seed>.+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop faces into per-character subfolders."
    )
    parser.add_argument("--image_dir", required=True, help="Folder with raw images.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination root folder for cropped images.",
    )
    return parser.parse_args()


def collect_images(input_dir: Path):
    return sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )


def parse_image_name(image_path: Path) -> Tuple[str, str]:
    match = NAME_PATTERN.match(image_path.stem)
    if not match:
        raise ValueError(
            f"Image name {image_path.name} does not match expected pattern 'character_[1-3]_..._X_seedY.ext'"
        )
    character_idx = match.group("char")
    x_value = match.group("x")
    return character_idx, x_value


def prepare_output_dirs(base_dir: Path) -> Dict[str, Path]:
    character_dirs = {}
    for character_name in CHARACTER_SUBFOLDERS:
        character_dir = base_dir / character_name
        character_dir.mkdir(parents=True, exist_ok=True)
        character_dirs[character_name] = character_dir
    return character_dirs


def main():
    args = parse_args()
    input_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Input path {input_dir} is not a directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path {output_dir} exists and is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"No supported image files found in {input_dir}")
        return

    character_dirs = prepare_output_dirs(output_dir)
    counters = {name: 0 for name in CHARACTER_SUBFOLDERS}
    landmarks_est = LandmarksEstimation(type="2D")

    for image_path in image_paths:
        try:
            character_idx, x_value = parse_image_name(image_path)
        except ValueError as err:
            print(f"Skipping {image_path.name}: {err}")
            continue

        character_key = f"character_{character_idx}"
        if character_key not in character_dirs:
            print(f"Skipping {image_path.name}: unexpected character '{character_idx}'")
            continue

        counters[character_key] += 1
        ordinal = counters[character_key]
        output_filename = f"{ordinal}_{x_value}{image_path.suffix.lower()}"
        save_path = character_dirs[character_key] / output_filename

        cropped_image = preprocess_image(
            str(image_path), landmarks_est, save_filename=str(save_path)
        )
        if cropped_image is None:
            print(f"Skipping {image_path.name} (no face detected).")
            counters[character_key] -= 1
            continue

        print(
            f"Cropped {image_path.name} -> {save_path.relative_to(output_dir)} ({cropped_image.shape})"
        )


if __name__ == "__main__":
    main()
