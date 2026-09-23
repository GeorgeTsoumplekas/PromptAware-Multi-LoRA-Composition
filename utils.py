import json
import itertools


def load_lora_info(image_style, path="lora_info.json"):
    path = f"{image_style}_{path}"
    with open(path) as f:
        lora_info = json.loads(f.read())
    return lora_info


def get_prompt(image_style):
    if image_style == "anime":
        prompt = "masterpiece, best quality"
        negative_prompt = "EasyNegative, extra fingers, extra limbs, fewer fingers, fewer limbs, multiple girls, multiple views, worst quality, low quality, depth of field, blurry, greyscale, 3D face, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, reference sheet, curvy, plump, fat, muscular female, strabismus, clothing cutout, side slit, tattoo, nsfw"
    else:
        prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up"
        negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt, negative_prompt


def generate_combinations(lora_info, compos_num):

    all_combinations = []
    elements = list(lora_info.keys())

    # All elements
    if compos_num == 1:
        selected_types = list(itertools.combinations(elements, 1))
        for types in selected_types:
            # Add 'character' to the current combination of types
            current_types = [*types]

            # Gather instances for each type in the current combination
            instances = [lora_info[t] for t in current_types]

            # Create combinations of instances across the selected types
            for combination in itertools.product(*instances):
                all_combinations.append(combination)

        return all_combinations

    # Elements restricted with 'character'
    else:
        # Check if the composition number is greater than the number of element types
        if compos_num > len(elements):
            raise ValueError(
                "The composition number cannot be greater than the number of elements."
            )

        # Ensure that 'character' is always included in the combinations
        if "character" in elements:
            # Remove 'character' from the list to avoid duplicating
            elements.remove("character")

            # Generate all possible combinations of the remaining element types
            selected_types = list(itertools.combinations(elements, compos_num - 1))

            # For each combination of types, generate all possible combinations of instances
            for types in selected_types:
                # Add 'character' to the current combination of types
                current_types = ["character", *types]

                # Gather instances for each type in the current combination
                instances = [lora_info[t] for t in current_types]

                # Create combinations of instances across the selected types
                for combination in itertools.product(*instances):
                    all_combinations.append(combination)

        return all_combinations
