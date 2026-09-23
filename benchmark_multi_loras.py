# Benchmark generation for W-Switch and W-Composite.

import argparse
import json
import os
import re
from pathlib import Path

from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
import torch
import torch.nn.functional as F

from callbacks import (
    DEFAULT_TYPE_BASED_WEIGHTS,
    make_weighted_callback,
)
from utils import generate_combinations, get_prompt


def build_type_weight_overrides(args):
    cli_values = {
        "character": args.weight_character,
        "clothing": args.weight_clothing,
        "style": args.weight_style,
        "background": args.weight_background,
        "object": args.weight_object,
        "other": args.weight_other,
    }

    overrides = {}
    for category, default_weight in DEFAULT_TYPE_BASED_WEIGHTS.items():
        value = cli_values.get(category)
        overrides[category] = default_weight if value is None else value
    return overrides


def infer_image_style_from_path(lora_path: str) -> str:
    parts = [part.lower() for part in Path(lora_path).parts if part]
    if "anime" in parts:
        return "anime"
    if "reality" in parts:
        return "reality"
    return "reality"


def resolve_lora_info_path(configured_path: str, image_style: str) -> str:
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


def get_lora_triggers(lora_info: dict, lora_id: str):
    for loras in lora_info.values():
        for lora in loras:
            if lora["id"] == lora_id:
                return lora.get("trigger", [])
    return []


def extend_prompt_with_triggers(prompt: str, lora_ids, lora_info: dict) -> str:
    if not lora_ids:
        return prompt

    trigger_phrases = []
    for lora_id in lora_ids:
        trigger_phrases.extend(get_lora_triggers(lora_info, lora_id))

    trigger_phrases = [t for t in trigger_phrases if t]
    if not trigger_phrases:
        return prompt

    suffix = ", ".join(trigger_phrases)
    return f"{prompt}, {suffix}" if prompt.strip() else suffix


def ensure_loras_loaded(pipeline, adapters, args, loaded_loras):
    for lora_id in adapters:
        if lora_id in loaded_loras:
            continue
        print(f"  Loading LoRA: {lora_id}")
        pipeline.load_lora_weights(
            args.lora_path, weight_name=f"{lora_id}.safetensors", adapter_name=lora_id
        )
        loaded_loras.add(lora_id)


def calculate_ablation_weights_ablated(
    pipeline, prompt, adapters, lora_info, device="cuda"
):
    print(f"Original prompt: {prompt}")

    full_prompt_inputs = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        full_prompt_embeds = pipeline.text_encoder(full_prompt_inputs.input_ids)[0]
        full_prompt_vector = full_prompt_embeds.mean(dim=1)

    weights = {}

    for adapter in adapters:
        triggers = []
        for category, loras in lora_info.items():
            for lora in loras:
                if lora["id"] == adapter:
                    triggers = lora["trigger"]
                    break
        if not triggers:
            weights[adapter] = 0.1
            continue

        prompt_without_triggers = prompt
        for trigger in triggers:
            prompt_without_triggers = re.sub(
                r"\b" + re.escape(trigger) + r"\b",
                "",
                prompt_without_triggers,
                flags=re.IGNORECASE,
            )

        prompt_without_triggers = re.sub(r"\s*,\s*,\s*", ", ", prompt_without_triggers)
        prompt_without_triggers = re.sub(r"\s+", " ", prompt_without_triggers).strip()
        prompt_without_triggers = prompt_without_triggers.strip(",").strip()

        print(f"Prompt without triggers: {prompt_without_triggers}")

        if not prompt_without_triggers or len(prompt_without_triggers) < 5:
            ablation_importance = 2.0
        else:
            ablated_inputs = pipeline.tokenizer(
                prompt_without_triggers,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                ablated_embeds = pipeline.text_encoder(ablated_inputs.input_ids)[0]
                ablated_vector = ablated_embeds.mean(dim=1)

            similarity = F.cosine_similarity(
                full_prompt_vector, ablated_vector, dim=-1
            ).item()

            ablation_importance = 1.0 - similarity

        weights[adapter] = max(ablation_importance, 0.05)

    total = sum(weights.values())
    if total > 0:
        weights = {k: (v / total) for k, v in weights.items()}

    return weights


def calculate_ablation_weights_triggers(
    pipeline, prompt, adapters, lora_info, device="cuda"
):
    print(f"Original prompt: {prompt}")

    full_prompt_inputs = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        full_prompt_embeds = pipeline.text_encoder(full_prompt_inputs.input_ids)[0]
        full_prompt_vector = full_prompt_embeds.mean(dim=1)

    weights = {}

    for adapter in adapters:
        triggers = []
        for category, loras in lora_info.items():
            for lora in loras:
                if lora["id"] == adapter:
                    triggers = lora["trigger"]
                    break

        trigger_word_prompt = ", ".join([t for t in triggers if t])

        if not trigger_word_prompt:
            weights[adapter] = 0.0
            continue

        print(f"Trigger word prompt: {trigger_word_prompt}")

        trigger_inputs = pipeline.tokenizer(
            trigger_word_prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            trigger_embeds = pipeline.text_encoder(trigger_inputs.input_ids)[0]
            trigger_vector = trigger_embeds.mean(dim=1)

        similarity = F.cosine_similarity(
            full_prompt_vector, trigger_vector, dim=-1
        ).item()

        weights[adapter] = similarity
        print(f"Weight for {adapter} before normalizations: {weights[adapter]:.3f}")

    total = sum(weights.values())
    if total > 0:
        weights = {k: (v / total) for k, v in weights.items()}

    print(f"Weights after normalizations: {weights}")

    return weights


def resolve_seed_list(args):
    if args.seeds is not None:
        if len(args.seeds) < args.num_images:
            raise ValueError(
                f"Provided {len(args.seeds)} seeds, but num_images={args.num_images}."
            )
        return args.seeds[: args.num_images]
    return [args.seed + i for i in range(args.num_images)]


def normalize_method_and_ablation(method: str, default_ablation: str):
    """
    Normalize method names that encode ablation strategy via suffix.
    Examples:
      weighted_switch_adaptive_ablated   -> (weighted_switch_adaptive, ablated)
      weighted_switch_adaptive_triggered -> (weighted_switch_adaptive, triggers)
    """
    ablation_method = default_ablation
    base_method = method

    if method.endswith("_ablated"):
        base_method = method.rsplit("_ablated", 1)[0]
        ablation_method = "ablated"
    elif method.endswith("_triggered"):
        base_method = method.rsplit("_triggered", 1)[0]
        ablation_method = "triggers"

    return base_method, ablation_method


def find_character_adapter(adapters):
    """Return the first adapter that looks like a character LoRA, if any."""
    for adapter in adapters:
        if adapter.lower().startswith("character"):
            return adapter
    return None


def generate_image(
    pipeline,
    prompt,
    negative_prompt,
    seed,
    save_path,
    method,
    ablation_method,
    adapters,
    args,
    lora_info,
):
    if not adapters:
        raise ValueError("At least one LoRA adapter is required.")

    switch_callback = None
    adaptive_weights = None
    method, ablation_method = normalize_method_and_ablation(method, ablation_method)

    requires_adaptive = "adaptive" in method

    if requires_adaptive:
        if ablation_method == "ablated":
            adaptive_weights = calculate_ablation_weights_ablated(
                pipeline=pipeline,
                prompt=prompt,
                adapters=adapters,
                lora_info=lora_info,
                device="cuda",
            )
        elif ablation_method == "triggers":
            adaptive_weights = calculate_ablation_weights_triggers(
                pipeline=pipeline,
                prompt=prompt,
                adapters=adapters,
                lora_info=lora_info,
                device="cuda",
            )
        else:
            raise ValueError(f"Unknown ablation method: {ablation_method}")
        print("  Prompt-adaptive weights:")
        for adapter, weight in adaptive_weights.items():
            print(f"    {adapter}: {weight:.3f}")

    if method == "weighted_composite_adaptive":
        pipeline.set_adapters(adapters)
        pipeline.set_lora_composition_weights(adaptive_weights)
    elif method in {"weighted_switch_adaptive", "weighted_switch_adaptive_tailed"}:
        pipeline.set_adapters([adapters[0]])
        weights_list = [adaptive_weights[lora] for lora in adapters]
        tail_reserved_steps = args.reserved_steps if "tailed" in method else 0
        tail_adapter = None
        if tail_reserved_steps:
            tail_adapter = find_character_adapter(adapters)
            if tail_adapter is None:
                raise ValueError(
                    "reserved_steps was set but no character LoRA was found in adapters."
                )
        switch_callback = make_weighted_callback(
            loras=adapters,
            weights=weights_list,
            num_inference_steps=args.denoise_steps,
            switch_step=args.switch_step,
            tail_reserved_steps=tail_reserved_steps or None,
            tail_adapter=tail_adapter,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print("\n  === LoRA Configuration ===")
    composite_methods = {"weighted_composite_adaptive"}
    if method in composite_methods:
        print(f"  Method: {method} (simultaneous LoRAs)")
        for adapter in adapters:
            weight = pipeline.get_lora_weight(adapter)
            print(f"    {adapter}: {weight:.3f}")

    generator = torch.manual_seed(seed)

    is_composite = method in composite_methods
    is_weighted = "weighted" in method
    use_lora_composite = is_composite and len(adapters) > 1
    use_weight_system = is_weighted and use_lora_composite

    print(
        f"  Pipeline config: lora_composite={use_lora_composite}, lora_composite_weighted={use_weight_system}"
    )

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=generator,
        cross_attention_kwargs={"scale": args.lora_scale},
        callback_on_step_end=switch_callback,
        lora_composite=use_lora_composite,
        lora_composite_weighted=use_weight_system,
    ).images[0]

    image.save(save_path)


def main(args):

    image_style = infer_image_style_from_path(args.lora_path)
    print(f"Inferred image style from LoRA path '{args.lora_path}': {image_style}")
    base_prompt, negative_prompt = get_prompt(image_style)
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = resolve_seed_list(args)

    model_name = (
        "gsdf/Counterfeit-V2.5"
        if image_style == "anime"
        else "SG161222/Realistic_Vision_V5.1_noVAE"
    )
    pipeline = DiffusionPipeline.from_pretrained(
        model_name, custom_pipeline="./pipelines/sd1.5_0.26.3", use_safetensors=True
    ).to("cuda")

    if image_style == "reality":
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        ).to("cuda")
        pipeline.vae = vae

    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    type_weight_overrides = build_type_weight_overrides(args)
    pipeline.configure_type_based_lora_weights(**type_weight_overrides)

    normalized_method, ablation_method = normalize_method_and_ablation(
        args.method,
        args.ablation_method,
    )
    if normalized_method != args.method:
        print(
            f"Normalized method '{args.method}' -> '{normalized_method}' (ablation={ablation_method})"
        )
    else:
        print(f"Using method '{normalized_method}' (ablation={ablation_method})")

    lora_info_path = resolve_lora_info_path(args.lora_info_file, image_style)
    with open(lora_info_path, "r") as f:
        lora_info = json.load(f)

    combos = generate_combinations(lora_info, args.composition_size)
    if not combos:
        raise ValueError("No LoRA combinations found with the provided settings.")
    print(f"Generating images for {len(combos)} LoRA combinations.")

    loras_per_combo = len(combos[0]) if combos else args.composition_size
    ablation_suffix = ""
    has_encoded_suffix = args.method.endswith("_ablated") or args.method.endswith(
        "_triggered"
    )
    if "adaptive" in normalized_method and not has_encoded_suffix:
        ablation_suffix = "_ablated" if ablation_method == "ablated" else "_triggered"

    method_output_dir = os.path.join(
        args.output_dir, f"{args.method}{ablation_suffix}_{loras_per_combo}"
    )
    os.makedirs(method_output_dir, exist_ok=True)

    loaded_loras = set()

    for combo_idx, combo in enumerate(combos, start=1):
        combo_lora_ids = [lora["id"] for lora in combo]
        prompt_with_triggers = extend_prompt_with_triggers(
            base_prompt, combo_lora_ids, lora_info
        )

        print(f"\n{'=' * 60}")
        print(f"Combination {combo_idx}/{len(combos)}")
        print(f"  LoRAs: {combo_lora_ids}")
        print(f"  Prompt: {prompt_with_triggers}")
        print(f"  Negative prompt: {negative_prompt}")

        ensure_loras_loaded(pipeline, combo_lora_ids, args, loaded_loras)

        lora_slug = "_".join(combo_lora_ids)

        for img_idx, seed in enumerate(seeds, start=1):
            filename = f"{lora_slug}_{img_idx}_seed{seed}.png"
            save_path = os.path.join(method_output_dir, filename)

            print(
                f"\n  Generating image {img_idx}/{args.num_images} for combo {combo_idx}: {save_path}"
            )
            generate_image(
                pipeline=pipeline,
                prompt=prompt_with_triggers,
                negative_prompt=negative_prompt,
                seed=seed,
                save_path=save_path,
                method=normalized_method,
                ablation_method=ablation_method,
                adapters=combo_lora_ids,
                args=args,
                lora_info=lora_info,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComposLORA benchmarking script")

    parser.add_argument(
        "--output_dir",
        default="benchmark_composlora",
        help="path to save the generated images",
        type=str,
    )
    parser.add_argument(
        "--seed", default=11, help="starting seed for generating images", type=int
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        type=int,
        help="explicit list of seeds to use for each generated image",
    )
    parser.add_argument(
        "--num_images",
        default=1,
        help="number of images to generate per prompt with different seeds",
        type=int,
    )
    parser.add_argument(
        "--composition_size",
        default=2,
        help="number of LoRAs to compose per combination",
        type=int,
    )
    parser.add_argument(
        "--ablation_method",
        default="ablated",
        choices=["ablated", "triggers"],
        help="method for computing adaptive weights",
        type=str,
    )
    parser.add_argument(
        "--lora_path",
        default="models/lora/reality",
        help="path to store all LoRAs",
        type=str,
    )
    parser.add_argument(
        "--lora_info_file",
        default="reality_lora_info.json",
        help="path to JSON file with LoRA trigger information",
        type=str,
    )
    parser.add_argument(
        "--lora_scale",
        default=0.8,
        help="scale of each LoRA when generating images",
        type=float,
    )
    parser.add_argument(
        "--weight_character",
        default=None,
        type=float,
        help="optional override for character LoRA weight (defaults to callbacks.DEFAULT_TYPE_BASED_WEIGHTS)",
    )
    parser.add_argument(
        "--weight_clothing",
        default=None,
        type=float,
        help="optional override for clothing LoRA weight (defaults to callbacks.DEFAULT_TYPE_BASED_WEIGHTS)",
    )
    parser.add_argument(
        "--weight_style",
        default=None,
        type=float,
        help="optional override for style LoRA weight (defaults to callbacks.DEFAULT_TYPE_BASED_WEIGHTS)",
    )
    parser.add_argument(
        "--weight_background",
        default=None,
        type=float,
        help="optional override for background LoRA weight (defaults to callbacks.DEFAULT_TYPE_BASED_WEIGHTS)",
    )
    parser.add_argument(
        "--weight_object",
        default=None,
        type=float,
        help="optional override for object LoRA weight (defaults to callbacks.DEFAULT_TYPE_BASED_WEIGHTS)",
    )
    parser.add_argument(
        "--weight_other",
        default=None,
        type=float,
        help="optional override for uncategorized LoRA weight (defaults to callbacks.DEFAULT_TYPE_BASED_WEIGHTS)",
    )
    parser.add_argument(
        "--method",
        default="weighted_switch_adaptive_tailed",
        choices=[
            "weighted_switch_adaptive_tailed",
            "weighted_switch_adaptive_tailed_ablated",
            "weighted_composite_adaptive",
            "weighted_composite_adaptive_triggered",
        ],
        help="W-Switch (PAW) or W-Composite (pass --ablation_method triggers for PTW)",
        type=str,
    )
    parser.add_argument(
        "--switch_step", default=5, help="base block length tau for W-Switch", type=int
    )
    parser.add_argument(
        "--reserved_steps",
        default=5,
        help="final denoising steps reserved for the character LoRA in W-Switch",
        type=int,
    )
    parser.add_argument(
        "--height", default=1024, help="height of the generated images", type=int
    )
    parser.add_argument(
        "--width", default=768, help="width of the generated images", type=int
    )
    parser.add_argument(
        "--denoise_steps", default=100, help="number of the denoising steps", type=int
    )
    parser.add_argument(
        "--cfg_scale", default=7, help="scale for classifier-free guidance", type=float
    )

    args = parser.parse_args()
    main(args)
