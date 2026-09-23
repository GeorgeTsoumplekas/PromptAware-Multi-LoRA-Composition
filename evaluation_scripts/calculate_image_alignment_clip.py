import argparse
import csv
import heapq
import os
from pathlib import Path
from typing import Dict, List

import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
TARGET_PREFIXES = ("background_", "character_", "clothing_", "object_", "style_")
STYLE_PREFIX = "style_"
NON_STYLE_SUFFIX = "_cropped"


def is_target_dir(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in TARGET_PREFIXES)


def collect_image_paths(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def load_and_encode_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(tensor).squeeze(0)
    return embedding.cpu().numpy()


def discover_generated_folders(root: Path) -> List[Path]:
    target_dirs: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        folder_name = os.path.basename(dirpath)
        if not is_target_dir(folder_name):
            continue
        if any(fname.lower().endswith(IMAGE_EXTENSIONS) for fname in filenames):
            target_dirs.append(Path(dirpath))
    return sorted(target_dirs)


def resolve_reference_dir(label: str, style_root: Path, cropped_root: Path) -> Path:
    if label.startswith(STYLE_PREFIX):
        return style_root / label
    return cropped_root / f"{label}{NON_STYLE_SUFFIX}"


def build_reference_cache(
    required_labels: List[str], style_root: Path, cropped_root: Path
) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for label in required_labels:
        reference_dir = resolve_reference_dir(label, style_root, cropped_root)
        if not reference_dir.is_dir():
            print(
                f"[WARN] Reference folder for '{label}' not found at {reference_dir}."
            )
            continue
        reference_images = collect_image_paths(reference_dir)
        if not reference_images:
            print(f"[WARN] No reference images found for '{label}' at {reference_dir}.")
            continue

        embeddings: List[np.ndarray] = []
        for image_path in reference_images:
            try:
                embeddings.append(load_and_encode_image(image_path))
            except Exception as err:  # pragma: no cover - best-effort logging
                print(f"[WARN] Failed to encode reference image {image_path}: {err}")

        if embeddings:
            cache[label] = {
                "embeddings": np.stack(embeddings, axis=0),
            }
        else:
            print(f"[WARN] All reference images failed to process for '{label}'.")
    return cache


def score_generated_folder(
    folder: Path,
    reference_embeddings: np.ndarray,
    method: str,
    top_k: int,
) -> List[float]:
    image_paths = collect_image_paths(folder)
    scores: List[float] = []

    if method == "topk" and top_k < 1:
        raise ValueError(
            "top_k must be a positive integer when using the 'topk' method."
        )

    for image_path in image_paths:
        try:
            embedding = load_and_encode_image(image_path)
        except Exception as err:  # pragma: no cover - best-effort logging
            print(f"[WARN] Failed to encode generated image {image_path}: {err}")
            continue

        similarities = cosine_similarity([embedding], reference_embeddings).astype(
            np.float32
        )
        per_image_sims = similarities.flatten().tolist()

        if method == "average":
            per_image_score = float(np.mean(per_image_sims))
        elif method == "max":
            per_image_score = float(np.max(per_image_sims))
        elif method == "topk":
            k = min(top_k, len(per_image_sims))
            top_scores = heapq.nlargest(k, per_image_sims)
            per_image_score = float(np.mean(top_scores))
        else:
            raise ValueError(f"Unsupported similarity method: {method}")

        scores.append(per_image_score)
    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate CLIP-based image alignment."
            "Each target subfolder's score is the mean similarity of its images against the "
            "corresponding reference concept folder."
        )
    )
    parser.add_argument(
        "--generated_root",
        required=True,
        help="Root folder under which generated images are stored.",
        type=str,
    )
    parser.add_argument(
        "--style_reference_root",
        default="concept_images",
        help="Root folder containing uncropped style concept images (style_1, style_2, ...).",
        type=str,
    )
    parser.add_argument(
        "--cropped_reference_root",
        default="concept_images_cropped",
        help="Root folder containing cropped concept images for background/character/clothing/object.",
        type=str,
    )
    parser.add_argument(
        "--similarity_method",
        choices=["average", "max", "topk"],
        default="average",
        help="strategy for per-image similarity aggregation",
        type=str,
    )
    parser.add_argument(
        "--top_k",
        default=1,
        type=int,
        help="k for 'topk' similarity (ignored for other methods)",
    )
    parser.add_argument(
        "--csv_filename",
        default="image_alignment_clip_results.csv",
        help="base CSV filename (method suffix is appended automatically)",
        type=str,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generated_root = Path(args.generated_root).expanduser().resolve()
    style_root = Path(args.style_reference_root).expanduser().resolve()
    cropped_root = Path(args.cropped_reference_root).expanduser().resolve()

    if not generated_root.is_dir():
        raise ValueError(f"Generated root {generated_root} is not a directory.")
    if not style_root.is_dir():
        raise ValueError(f"Style reference root {style_root} is not a directory.")
    if not cropped_root.is_dir():
        raise ValueError(f"Cropped reference root {cropped_root} is not a directory.")

    target_folders = discover_generated_folders(generated_root)
    if not target_folders:
        raise ValueError(
            f"No target folders (background_*, character_*, ...) found under {generated_root}."
        )

    required_labels = sorted({folder.name for folder in target_folders})
    reference_cache = build_reference_cache(required_labels, style_root, cropped_root)

    if not reference_cache:
        raise ValueError("Reference cache is empty. Unable to score generated folders.")

    base_csv_path = Path(args.csv_filename)
    suffix = base_csv_path.suffix or ".csv"
    method_suffix = (
        f"{args.similarity_method}_k{args.top_k}"
        if args.similarity_method == "topk"
        else args.similarity_method
    )
    csv_filename = f"{base_csv_path.stem}_{method_suffix}{suffix}"
    csv_path = generated_root / csv_filename

    overall_scores: List[float] = []
    rows = []

    for folder in target_folders:
        label = folder.name
        if label not in reference_cache:
            print(
                f"[WARN] Skipping {folder}: missing reference embeddings for {label}."
            )
            continue

        reference_data = reference_cache[label]
        image_scores = score_generated_folder(
            folder,
            reference_data["embeddings"],
            args.similarity_method,
            args.top_k,
        )
        if not image_scores:
            print(f"[WARN] Folder {folder} contains no valid generated images.")
            continue

        folder_avg = float(np.mean(image_scores))
        overall_scores.extend(image_scores)
        rows.append(
            {
                "folder_path": str(folder),
                "folder_label": label,
                "num_images": len(image_scores),
                "average_similarity": folder_avg,
                "method": args.similarity_method,
            }
        )
        print(
            f"[INFO] {label}: {len(image_scores)} images, "
            f"average CLIP similarity = {folder_avg:.4f}"
        )

    if not rows:
        raise ValueError("No generated folders produced valid similarity scores.")

    overall_average = float(np.mean(overall_scores))
    print(f"[RESULT] Overall average CLIP similarity: {overall_average:.4f}")

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "folder_path",
                "folder_label",
                "num_images",
                "average_similarity",
                "method",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "folder_path": "OVERALL",
                "folder_label": "",
                "num_images": len(overall_scores),
                "average_similarity": overall_average,
                "method": args.similarity_method,
            }
        )


if __name__ == "__main__":
    main()
