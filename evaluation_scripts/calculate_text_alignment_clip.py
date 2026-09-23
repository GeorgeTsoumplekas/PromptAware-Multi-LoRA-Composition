import argparse
import csv
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import get_prompt


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def infer_image_style_from_path(lora_path: str) -> str:
    """
    Heuristic used in benchmark_small_scale_multi_loras.py to pick prompt defaults.
    If the LoRA path contains "anime" we treat it as anime, otherwise reality.
    """
    parts = [part.lower() for part in Path(lora_path).parts if part]
    if "anime" in parts:
        return "anime"
    if "reality" in parts:
        return "reality"
    return "reality"


def resolve_lora_info_path(configured_path: str, image_style: str) -> str:
    """
    Mirror of benchmark_small_scale_multi_loras.resolve_lora_info_path.
    Tries a style-prefixed variant before falling back to the provided path.
    """
    path = Path(configured_path)
    if image_style in path.name and path.is_file():
        return str(path)

    suffix = path.name.split("_", 1)[1] if "_" in path.name else path.name
    styled_candidate = path.with_name(f"{image_style}_{suffix}")
    if styled_candidate.is_file():
        return str(styled_candidate)

    if path.is_file():
        return str(path)

    raise FileNotFoundError(
        f"Could not find LoRA info file for style '{image_style}'. "
        f"Tried '{path}' and '{styled_candidate}'."
    )


def extend_prompt_with_triggers(
    prompt: str, lora_ids: Sequence[str], lora_info: Dict
) -> str:
    """
    Append trigger phrases for the provided LoRA ids to the base prompt.
    Matches benchmark_small_scale_multi_loras.extend_prompt_with_triggers.
    """
    if not lora_ids:
        return prompt

    trigger_phrases: List[str] = []
    for lora_id in lora_ids:
        for loras in lora_info.values():
            for lora in loras:
                if lora.get("id") == lora_id:
                    trigger_phrases.extend(lora.get("trigger", []))
                    break

    trigger_phrases = [t for t in trigger_phrases if t]
    if not trigger_phrases:
        return prompt

    suffix = ", ".join(trigger_phrases)
    return f"{prompt}, {suffix}" if prompt.strip() else suffix


def load_and_preprocess_image(image_path: str):
    """Load and preprocess an image for CLIP."""
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)


def encode_text_cached(prompt: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Encode text using CLIP with a simple cache to avoid duplicate work."""
    if prompt in cache:
        return cache[prompt]

    try:
        text_tokens = clip.tokenize([prompt]).to(device)
    except RuntimeError as e:
        # Some prompts become too long once LoRA triggers are appended; fall back to truncation.
        if "too long for context length" in str(e):
            print(f"Warning: prompt too long for CLIP; truncating: {prompt}")
            text_tokens = clip.tokenize([prompt], truncate=True).to(device)
        else:
            raise
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).squeeze(0)
    embedding = text_embedding.cpu().numpy()
    cache[prompt] = embedding
    return embedding


def encode_image(image_path: str) -> np.ndarray:
    """Encode image using CLIP."""
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        image_embedding = model.encode_image(image).squeeze(0)
    return image_embedding.cpu().numpy()


def collect_lora_ids(lora_info: Dict) -> List[str]:
    """Flatten LoRA ids from the info json."""
    ids = []
    for loras in lora_info.values():
        for lora in loras:
            if "id" in lora:
                ids.append(lora["id"])
    return ids


def parse_image_metadata(filename: str) -> Optional[Tuple[str, int, int]]:
    """
    Extract the LoRA slug, image index and seed from a filename like:
        <slug>_<imgIdx>_seed<seed>.png
    """
    stem = Path(filename).stem
    match = re.match(r"(?P<slug>.+?)_(?P<img_idx>\d+)_seed(?P<seed>-?\d+)$", stem)
    if not match:
        return None
    return match.group("slug"), int(match.group("img_idx")), int(match.group("seed"))


def build_slug_splitter(valid_ids: Iterable[str]):
    """
    Create a splitter function that decomposes a slug (ids joined by "_")
    back into the original LoRA id list using backtracking.
    """
    valid_set = set(valid_ids)

    @lru_cache(maxsize=None)
    def split(slug: str) -> List[List[str]]:
        if not slug:
            return [[]]

        candidates: List[List[str]] = []
        for lora_id in valid_set:
            if slug.startswith(lora_id):
                remainder = slug[len(lora_id) :]
                if remainder and remainder[0] != "_":
                    continue
                next_slug = remainder[1:] if remainder.startswith("_") else remainder
                for tail in split(next_slug):
                    candidates.append([lora_id] + tail)
        return candidates

    def splitter(slug: str) -> Optional[List[str]]:
        all_splits = split(slug)
        if not all_splits:
            return None
        # Prefer the split with the most LoRAs (finer granularity); tie-break by longest total length.
        all_splits.sort(key=lambda parts: (-len(parts), -sum(len(p) for p in parts)))
        best = all_splits[0]
        if len(all_splits) > 1:
            print(
                f"Warning: Multiple segmentations for slug '{slug}'. "
                f"Choosing {best} from {all_splits}."
            )
        return best

    return splitter


def calculate_text_image_alignment(
    image_folder: str, base_prompt: str, lora_info: Dict
):
    """
    Calculate text-image alignment for ComposLoRA outputs.
    Reconstructs prompts from image filenames by extracting LoRA ids from the slug.
    """
    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not image_files:
        print("No images found in the provided folder.")
        return []

    valid_ids = collect_lora_ids(lora_info)
    split_slug = build_slug_splitter(valid_ids)

    results = []
    text_cache: Dict[str, np.ndarray] = {}

    for image_file in sorted(image_files):
        parsed = parse_image_metadata(image_file)
        if not parsed:
            print(
                f"Warning: Filename '{image_file}' does not match expected pattern "
                f"'<slug>_<idx>_seed<seed>.png'. Skipping."
            )
            continue

        slug, img_idx, seed = parsed
        lora_ids = split_slug(slug)
        if not lora_ids:
            print(
                f"Warning: Could not map slug '{slug}' to known LoRA ids. Skipping '{image_file}'."
            )
            continue

        prompt = extend_prompt_with_triggers(base_prompt, lora_ids, lora_info)
        text_embedding = encode_text_cached(prompt, text_cache)

        image_path = os.path.join(image_folder, image_file)
        image_embedding = encode_image(image_path)

        similarity = cosine_similarity([text_embedding], [image_embedding])[0][0]

        results.append(
            {
                "image_file": image_file,
                "image_index": img_idx,
                "seed": seed,
                "lora_ids": lora_ids,
                "prompt": prompt,
                "similarity": similarity,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate text-to-image alignment for ComposLoRA outputs using CLIP."
    )
    parser.add_argument(
        "--generated_image_folder",
        required=True,
        type=str,
        help="Path to the folder containing generated images named <slug>_<idx>_seed<seed>.png",
    )
    parser.add_argument(
        "--lora_info_file",
        default="reality_lora_info.json",
        type=str,
        help="Path to JSON file with LoRA trigger information. Style-prefixed variants are discovered automatically.",
    )
    parser.add_argument(
        "--lora_path",
        default="models/lora/reality",
        type=str,
        help="Path used to infer image style (anime vs reality), mirroring benchmark_small_scale_multi_loras.py",
    )
    parser.add_argument(
        "--image_style",
        default=None,
        type=str,
        help="Override inferred style (anime|reality). If omitted, inferred from --lora_path.",
    )
    parser.add_argument(
        "--output_csv",
        default="text_alignment_clip_composlora_results.csv",
        type=str,
        help="Output CSV filename (written inside generated_image_folder).",
    )

    args = parser.parse_args()

    image_style = args.image_style or infer_image_style_from_path(args.lora_path)
    base_prompt, _ = get_prompt(image_style)

    lora_info_path = resolve_lora_info_path(args.lora_info_file, image_style)
    with open(lora_info_path, "r") as f:
        lora_info = json.load(f)

    print(f"Using image style: {image_style}")
    print(f"Base prompt: {base_prompt}")
    print(f"Loaded LoRA info from: {lora_info_path}")
    print(f"Scanning images in: {args.generated_image_folder}")

    results = calculate_text_image_alignment(
        image_folder=args.generated_image_folder,
        base_prompt=base_prompt,
        lora_info=lora_info,
    )

    similarities = [r["similarity"] for r in results]
    if similarities:
        avg_similarity = float(np.mean(similarities))
        print(f"\n{'=' * 60}")
        print(f"Average Text-to-Image Similarity: {avg_similarity:.4f}")
        print(f"{'=' * 60}")
    else:
        avg_similarity = float("nan")
        print("\nNo valid images were processed. Please verify your inputs.")

    results_csv_path = os.path.join(args.generated_image_folder, args.output_csv)
    with open(results_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Image File",
                "Image Index",
                "Seed",
                "LoRA IDs",
                "Prompt",
                "Similarity",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["image_file"],
                    result["image_index"],
                    result["seed"],
                    ";".join(result["lora_ids"]),
                    result["prompt"],
                    result["similarity"],
                ]
            )
        writer.writerow([])
        writer.writerow(["Average", "", "", "", "", avg_similarity])


if __name__ == "__main__":
    main()
