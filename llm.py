import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
from os.path import join
import argparse
import re
from collections import defaultdict
from utils import load_lora_info, generate_combinations
from utils import get_prompt
import json

torch.manual_seed(0)


def main(args):
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )  # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V-2_6", trust_remote_code=True
    )

    args.lora_path = join(args.lora_path, args.image_style)
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    combinations = generate_combinations(lora_info, args.compos_num)
    init_prompt, _ = get_prompt(args.image_style)
    seeds = args.seeds

    # Provide the specific evluation senario and criteria
    task_description = "1) Element Integration:\nHow seamlessly different elements are combined within the image.\n\nCriteria:\n- Visual Cohesion: Evaluate whether the elements appear as part of a unified scene, rather than as disjointed parts.\n- Object Overlap and Interaction: Check for natural overlaps and interactions between objects, ensuring no awkward placements or intersections.\n\n2) Spatial Consistency:\nUniformity in style, lighting, and perspective across all elements.\n\nCriteria:\n- Stylistic Uniformity: Ensure that all elements share a consistent artistic style (e.g., realism, cartoonish).\n- Lighting and Shadows: Verify that light sources and shadow directions are consistent, contributing to a realistic portrayal.\n- Perspective Alignment: Confirm that elements adhere to a shared perspective, with no mismatched viewpoints.\n\n3) Semantic Accuracy:\nCorrect interpretation and representation of each element as described in the prompt.\n\nCriteria:\n- Object Accuracy: Objects should align with their descriptions in terms of type, attributes, and context.\n- Action and Interaction: Actions or interactions between objects should be depicted accurately and appropriately.\n\n4) Aesthetic Quality:\nOverall visual appeal and artistic quality of the generated image.\n\nCriteria:\n- Color Harmony: The use of color palettes should be visually pleasing and fitting for the scene.\n- Composition Balance: Elements should be arranged in a balanced way to create an engaging and harmonious composition.\n- Clarity and Sharpness: The image should be clear, with well-defined elements, free from unwanted blurriness or distortion."
    # provide more specific task description
    advance_task_description = ""

    # Prepare method directories
    method_dirs = []
    for method, base_path in zip(args.methods, args.paths):
        method_dir = os.path.join(
            base_path,
            f"{method}_{args.ablation_method}_{args.compos_num}"
            if args.ablation_method is not None
            else f"{method}_{args.compos_num}",
        )
        method_dirs.append(method_dir)

    metrics = ["integration", "consistency", "accuracy", "appeal"]
    totals = {m: defaultdict(float) for m in args.methods}
    counts = {m: 0 for m in args.methods}

    def extract_scores(answer_text):
        """
        Extract per-method scores assuming the model follows the requested JSON array:
        [
          {"method": "<name>", "integration": x, "consistency": y, "accuracy": z, "appeal": w},
          ...
        ]
        Only accepts numeric values in [0, 10].
        """

        def _coerce_to_str(payload):
            if isinstance(payload, dict):
                for key in ("text", "response", "content", "message"):
                    if key in payload:
                        return str(payload[key])
                return json.dumps(payload)
            return str(payload)

        def _parse_score(val):
            """Return float in [0,10] or None."""
            try:
                score = float(val)
            except (TypeError, ValueError):
                return None
            if 0.0 <= score <= 10.0:
                return score
            return None

        answer_text = _coerce_to_str(answer_text)

        # Try to parse JSON array first
        match = re.search(r"\[\s*{[\s\S]*}\s*\]", answer_text)
        json_blob = match.group(0) if match else answer_text

        try:
            data = json.loads(json_blob)
        except Exception:
            data = None

        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        if isinstance(data, list):
            parsed = []
            for entry in data:
                method = entry.get("method")
                if method is None:
                    continue
                scores = (
                    _parse_score(entry.get("integration")),
                    _parse_score(entry.get("consistency")),
                    _parse_score(entry.get("accuracy")),
                    _parse_score(entry.get("appeal")),
                )
                if any(s is None for s in scores):
                    continue
                parsed.append((method, scores))
            if parsed:
                return parsed

        # Fallback: parse plaintext blocks like "1. Method: X ... Integration: 9 ..."
        parsed = []
        block_re = re.compile(
            r"(?m)^\s*\d*\.?\s*Method:\s*(?P<method>[^\n]+)(?P<body>.*?)(?=^\s*\d*\.?\s*Method:|\Z)",
            re.S | re.M,
        )
        for match in block_re.finditer(answer_text):
            method = match.group("method").strip()
            body = match.group("body")
            scores = []
            for metric_name in ("Integration", "Consistency", "Accuracy", "Appeal"):
                m = re.search(
                    rf"{metric_name}:\s*(-?\d+(?:\.\d+)?)", body, re.IGNORECASE
                )
                if not m:
                    scores = []
                    break
                val = _parse_score(m.group(1))
                if val is None:
                    scores = []
                    break
                scores.append(val)
            if scores:
                parsed.append((method, tuple(scores)))
        return parsed

    def accumulate(answer_text):
        parsed = extract_scores(answer_text)
        print(f"Parsed scores: {parsed}")
        for method, vals in parsed:
            if method not in totals:
                continue
            for k, v in zip(metrics, vals):
                totals[method][k] += v
                print(f"Total {method} {k}: {totals[method][k]}")
            counts[method] += 1
        print(f"Counts so far: {dict(counts)}")

    for seed_idx, seed in enumerate(seeds, start=1):
        for combo in range(len(combinations)):
            triggers = [
                trigger for lora in combinations[combo] for trigger in lora["trigger"]
            ]
            prompt = init_prompt + ", " + ", ".join(triggers)
            lora_slug = "_".join([lora["id"] for lora in combinations[combo]])

            images = []
            for method_dir in method_dirs:
                file_name = f"{lora_slug}_{seed_idx}_seed{seed}.png"
                img_path = os.path.join(method_dir, file_name)
                images.append(Image.open(img_path).convert("RGB"))

            question = f'You are given {len(images)} images generated by different text-to-image methods. The expected concepts in the image include: {prompt}. Key attributes: {task_description}. {advance_task_description} \
                        Please rate each image independently on: Integration of Elements, Consistency in Composition, Accuracy in Depiction, Visual Appeal. \
                        Respond ONLY with a JSON array where each entry has: "method", "integration", "consistency", "accuracy", "appeal". Use numeric scores out of 10. The numbers can be any value between 0 and 10, and you may use both integer and .5 scores, for example: 5, 5.5, 7.5. No extra text.'

            msgs = [{"role": "user", "content": images + [question]}]

            answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            accumulate(answer)

    # Write average scores per method
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w") as f:
        for method in args.methods:
            cnt = max(1, counts[method])  # avoid div by zero
            f.write(f"Method: {method}\n")
            total_avg_sum = 0.0
            for k in metrics:
                avg = totals[method][k] / cnt if cnt else 0.0
                f.write(f"  {k}: {avg:.3f}\n")
                total_avg_sum += avg
            overall_avg = total_avg_sum / len(metrics) if metrics else 0.0
            f.write(f"  overall_avg: {overall_avg:.3f}\n")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given LoRAs in the ComposLoRA benchmark, generate images with arbitrary combinations based on LoRAs"
    )

    # Arguments for composing LoRAs
    parser.add_argument(
        "--compos_num",
        default=2,
        help="number of elements to be combined in a single image",
        type=int,
    )
    parser.add_argument(
        "--ablation_method",
        default=None,
        choices=["ablated", "triggers"],
        help="method for computing adaptive weights",
        type=str,
    )
    parser.add_argument(
        "--lora_path",
        default="models/lora",
        help="path to store all LoRA models",
        type=str,
    )
    parser.add_argument(
        "--lora_info_path",
        default="lora_info.json",
        help="path to store all LoRA information",
        type=str,
    )
    parser.add_argument(
        "--image_style",
        default="reality",
        choices=["anime", "reality"],
        help="sytles of the generated images",
        type=str,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="list of method names, order corresponds to paths",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="list of base folders for each method (same order as methods)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[1, 11, 111],
        type=int,
        help="space-separated list of seeds used to locate images",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="path to write aggregated average scores; defaults to llm_evals/minicpm_evaluation_{compos_num}.txt",
    )

    args = parser.parse_args()
    if len(args.methods) != len(args.paths):
        raise ValueError("methods and paths must have the same length")
    if args.output_txt is None:
        args.output_txt = os.path.join(
            "llm_evals", f"minicpm_evaluation_{args.compos_num}.txt"
        )

    main(args)
