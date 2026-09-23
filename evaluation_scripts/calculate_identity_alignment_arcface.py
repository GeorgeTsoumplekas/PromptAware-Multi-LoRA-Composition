import argparse
import csv
import heapq
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PRESERVATION_DIR = PROJECT_ROOT / "identity_preservation"
if str(IDENTITY_PRESERVATION_DIR) not in sys.path:
    sys.path.append(str(IDENTITY_PRESERVATION_DIR))


def load_id_loss_class():
    module_path = IDENTITY_PRESERVATION_DIR / "id_loss.py"
    spec = importlib.util.spec_from_file_location("id_loss", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IDLoss module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IDLoss


IDLoss = load_id_loss_class()


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
MAX_IMAGE_HEIGHT = 256
CHARACTER_SUBFOLDERS = ("character_1", "character_2", "character_3")


def image_resize(image, target_height):
    height, width = image.shape[:2]
    scale = target_height / float(height)
    new_width = max(1, int(round(width * scale)))
    resized = cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    return resized, scale


def load_image_tensor(image_source, device):
    if isinstance(image_source, (str, Path)) and os.path.isfile(image_source):
        image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image '{image_source}'")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")
    elif isinstance(image_source, np.ndarray):
        image = image_source
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    if image.shape[0] > MAX_IMAGE_HEIGHT:
        image, _ = image_resize(image, MAX_IMAGE_HEIGHT)

    tensor = torch.tensor(np.transpose(image, (2, 0, 1))).float().div(255.0)
    tensor = tensor * 2.0 - 1.0  # scale to [-1, 1]
    return tensor.unsqueeze(0).to(device)


def list_image_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return []
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_image_tensors(folder_path, device):
    """Load images as (path, tensor) tuples."""
    tensors = []
    for image_path in list_image_files(folder_path):
        try:
            image_tensor = load_image_tensor(image_path, device)
        except Exception as exc:
            print(f"Skipping '{image_path}' due to error: {exc}")
            continue
        tensors.append((image_path, image_tensor))
    return tensors


def calculate_similarity_for_pair(
    real_folder,
    generated_folder,
    id_module,
    device,
    crop,
    method,
    top_k,
):
    real_tensors = load_image_tensors(real_folder, device)
    generated_tensors = load_image_tensors(generated_folder, device)

    if not real_tensors or not generated_tensors:
        print(
            f"Skipping pair '{real_folder}' <-> '{generated_folder}' "
            "because one of the folders is empty or unreadable."
        )
        return [], 0

    if method == "topk" and top_k < 1:
        raise ValueError(
            "top_k must be a positive integer when using the 'topk' method."
        )

    similarity_scores = []  # per-generated-image similarity values
    for _, gen_tensor in generated_tensors:
        per_image_sims = []
        for _, real_tensor in real_tensors:
            with torch.no_grad():
                sim = id_module(gen_tensor, real_tensor, crop=crop)
            per_image_sims.append(sim.mean().item())

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

        similarity_scores.append(per_image_score)
    return similarity_scores, len(generated_tensors)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate ArcFace identity alignment across three characters"
    )
    parser.add_argument(
        "--real_image_folder",
        default="real_images",
        help="root folder containing reference images in character_{1,2,3} subfolders",
        type=str,
    )
    parser.add_argument(
        "--generated_image_folder",
        default="generated_images",
        help="root folder containing generated images in character_{1,2,3} subfolders",
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
        default="identity_alignment_arcface_composlora_results.csv",
        help="base CSV filename (method suffix is appended automatically)",
        type=str,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_module = IDLoss().to(device)
    id_module.eval()

    overall_similarity_scores = []
    total_generated_images = 0
    for character in CHARACTER_SUBFOLDERS:
        real_folder = Path(args.real_image_folder) / f"{character}_cropped"
        generated_folder = Path(args.generated_image_folder) / character

        if not real_folder.exists():
            print(f"Reference folder missing, skipping: {real_folder}")
            continue
        if not generated_folder.exists():
            print(f"Generated folder missing, skipping: {generated_folder}")
            continue

        similarity_scores, num_gen = calculate_similarity_for_pair(
            real_folder,
            generated_folder,
            id_module,
            device,
            False,
            args.similarity_method,
            args.top_k,
        )
        overall_similarity_scores.extend(similarity_scores)
        total_generated_images += num_gen

    if not overall_similarity_scores:
        print("No similarities computed; all folders were missing or empty.")
        return

    average_similarity = float(np.mean(overall_similarity_scores))
    print(f"Overall average cosine similarity: {average_similarity}")
    print(f"Total generated images examined: {total_generated_images}")

    os.makedirs(args.generated_image_folder, exist_ok=True)

    base_csv_path = Path(args.csv_filename)
    suffix = base_csv_path.suffix or ".csv"
    method_suffix = (
        f"{args.similarity_method}_k{args.top_k}"
        if args.similarity_method == "topk"
        else args.similarity_method
    )
    csv_filename = f"{base_csv_path.stem}_{method_suffix}{suffix}"
    csv_path = os.path.join(args.generated_image_folder, csv_filename)
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value", "NumSamples", "Method"])
        writer.writerow(
            [
                "overall_average_similarity",
                average_similarity,
                len(overall_similarity_scores),
                args.similarity_method,
            ]
        )


if __name__ == "__main__":
    main()
