import numpy as np

DEFAULT_TYPE_BASED_WEIGHTS = {
    "character": 1.0,
    "clothing": 1.0,
    "style": 1.0,
    "background": 1.0,
    "object": 1.0,
    "other": 1.0,
}


def create_weighted_schedule(loras, weights, num_steps, switch_step):
    """
    Create a schedule that assigns consecutive blocks of steps to each lora based on normalized weights.
    Similar to the original switch method but with weighted block sizes.

    Args:
        loras: List of lora adapter names
        weights: List of weights for each lora (will be normalized)
        num_steps: Total number of inference steps

    Returns:
        List of lora names, one for each step (length = num_steps)
    """
    if switch_step is None or switch_step <= 0:
        raise ValueError(
            "switch_step must be provided and > 0 for weighted schedule generation."
        )

    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        raise ValueError(
            "At least one weight must be positive to create a weighted schedule."
        )

    # Normalize relative to the maximum so equal weights map to identical block sizes.
    normalized = weights / weights.max()
    block_sizes = np.maximum(1, np.round(normalized * switch_step)).astype(int)

    schedule = []
    adapter_idx = 0
    while len(schedule) < num_steps:
        lora = loras[adapter_idx % len(loras)]
        block = block_sizes[adapter_idx % len(loras)]
        schedule.extend([lora] * block)
        adapter_idx += 1

    return schedule[:num_steps]  # Ensure exact length


def make_weighted_callback(
    loras,
    weights,
    num_inference_steps=50,
    switch_step=None,
    tail_reserved_steps=None,
    tail_adapter=None,
):
    """
    Weighted lora switch callback that activates each lora for a percentage of steps
    proportional to its weight. Loras are periodically rotated throughout the process.

    Args:
        loras: List of lora adapter names
        weights: List of weights for each lora (will be normalized to percentages).
        num_inference_steps: Total number of inference steps (needed to create schedule)
        tail_reserved_steps: Optional int; if provided, the final N steps will force-activate
                             the specified tail_adapter.
        tail_adapter: Optional adapter name to force during the tail portion.

    Example:
        With loras=['char1', 'char2', 'style1'] and weights=[0.5, 0.3, 0.2],
        char1 will be active for 50% of steps, char2 for 30%, and style1 for 20%.
    """
    # Validate inputs
    if len(loras) != len(weights):
        raise ValueError(
            f"Number of loras ({len(loras)}) must match number of weights ({len(weights)})"
        )

    # Create the schedule
    schedule = create_weighted_schedule(
        loras, weights, num_inference_steps, switch_step
    )

    # Optionally force the tail steps to a specific adapter (e.g., character LoRA)
    if tail_reserved_steps is not None and tail_reserved_steps > 0:
        if tail_adapter is None:
            raise ValueError(
                "tail_adapter must be provided when tail_reserved_steps is set."
            )
        if tail_reserved_steps > num_inference_steps:
            raise ValueError("tail_reserved_steps cannot exceed num_inference_steps.")
        if tail_adapter not in loras:
            raise ValueError(f"tail_adapter '{tail_adapter}' must be one of: {loras}")
        reserved = min(int(tail_reserved_steps), num_inference_steps)
        schedule[-reserved:] = [tail_adapter] * reserved

    # Debug: Print schedule summary
    print("\n=== Weighted LoRA Schedule ===")
    print(f"Total steps: {num_inference_steps}")
    for lora, weight in zip(loras, weights):
        count = schedule.count(lora)
        percentage = (count / num_inference_steps) * 100
        print(
            f"  {lora}: weight={weight:.2f}, steps={count}/{num_inference_steps} ({percentage:.1f}%)"
        )

    # Create callback function
    def weighted_switch_callback(pipeline, step_index, _timestep, _callback_kwargs):
        callback_outputs = {}

        # Set the appropriate lora for this step
        if step_index < len(schedule):
            target_lora = schedule[step_index]
            current_adapters = pipeline.get_active_adapters()

            # Only switch if we need to change lora
            if not current_adapters or current_adapters[0] != target_lora:
                pipeline.set_adapters([target_lora])

        return callback_outputs

    return weighted_switch_callback
