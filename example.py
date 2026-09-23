"""Single-image inference for W-Switch and W-Composite."""

import argparse
import json
import os
from pathlib import Path

from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler

from benchmark_multi_loras import (
    build_type_weight_overrides,
    ensure_loras_loaded,
    extend_prompt_with_triggers,
    generate_image,
    infer_image_style_from_path,
    resolve_lora_info_path,
)
from utils import get_prompt


REPO_ROOT = Path(__file__).resolve().parent

METHOD_SETTINGS = {
    "w-switch": {
        "method": "weighted_switch_adaptive_tailed",
        "ablation_method": "ablated",
        "reserved_steps": 5,
    },
    "w-composite": {
        "method": "weighted_composite_adaptive",
        "ablation_method": "triggers",
        "reserved_steps": 0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one image with W-Switch or W-Composite."
    )
    parser.add_argument(
        "--method",
        default="w-switch",
        choices=sorted(METHOD_SETTINGS),
        help="W-Switch uses PAW and reserves the last 5 steps for the character LoRA. "
        "W-Composite uses PTW.",
    )
    parser.add_argument(
        "--loras",
        nargs="+",
        default=["character_2", "clothing_2"],
        help="LoRA ids to compose. W-Switch expects one id that starts with 'character'.",
    )
    parser.add_argument(
        "--lora_path",
        default="models/lora/reality",
        help="Directory that contains <lora_id>.safetensors.",
    )
    parser.add_argument(
        "--lora_info_file",
        default="reality_lora_info.json",
        help="JSON file with the trigger words for each LoRA.",
    )
    parser.add_argument("--save_path", default=None, help="Output image path.")
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=768, type=int)
    parser.add_argument("--denoise_steps", default=100, type=int)
    parser.add_argument("--cfg_scale", default=7.0, type=float)
    parser.add_argument("--lora_scale", default=0.8, type=float)
    parser.add_argument("--switch_step", default=5, type=int)
    return parser.parse_args()


def load_pipeline(lora_path):
    image_style = infer_image_style_from_path(lora_path)
    model_name = (
        "gsdf/Counterfeit-V2.5"
        if image_style == "anime"
        else "SG161222/Realistic_Vision_V5.1_noVAE"
    )
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True,
    ).to("cuda")

    if image_style == "reality":
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")
        pipeline.vae = vae

    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)
    return pipeline, image_style


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    settings = METHOD_SETTINGS[args.method]
    method_name = args.method
    args.method = settings["method"]
    args.ablation_method = settings["ablation_method"]
    args.reserved_steps = settings["reserved_steps"]
    args.weight_character = None
    args.weight_clothing = None
    args.weight_style = None
    args.weight_background = None
    args.weight_object = None
    args.weight_other = None

    pipeline, image_style = load_pipeline(args.lora_path)
    type_weight_overrides = build_type_weight_overrides(args)
    pipeline.configure_type_based_lora_weights(**type_weight_overrides)

    lora_info_path = resolve_lora_info_path(args.lora_info_file, image_style)
    with open(lora_info_path, "r") as handle:
        lora_info = json.load(handle)

    base_prompt, negative_prompt = get_prompt(image_style)
    prompt = extend_prompt_with_triggers(base_prompt, args.loras, lora_info)
    print(f"Method: {args.method} ({args.ablation_method})")
    print(f"Prompt: {prompt}")

    ensure_loras_loaded(pipeline, args.loras, args, set())

    if args.save_path is None:
        slug = "_".join(args.loras)
        args.save_path = f"outputs/examples/{method_name}/{slug}_1_seed{args.seed}.png"
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    generate_image(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=args.seed,
        save_path=str(save_path),
        method=args.method,
        ablation_method=args.ablation_method,
        adapters=args.loras,
        args=args,
        lora_info=lora_info,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
