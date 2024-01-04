from typing import Dict
import torch


def interpolate_weights(
    model1_state_dict: Dict,
    model2_state_dict: Dict,
    alpha: float,
) -> dict:
    """
    Interpolate between two model state dicts and save the interpolated model to the saving_dir.
    Args:
        model_state_dict_1: The state dict of the first model
        model_state_dict_2: The state dict of the second model
        alpha: The interpolation factor
        saving_dir: The directory to save the interpolated model
    Returns:
        The interpolated model state dict
    """
    new_model_state_dict = {}
    with torch.no_grad():
        for param_name in model1_state_dict:
            assert param_name in model2_state_dict, f"Model structure mismatch: {param_name}"
            param1 = model1_state_dict[param_name].to("cpu")
            param2 = model2_state_dict[param_name].to("cpu")
            new_model_state_dict[param_name] = (
                alpha * param1 + (1 - alpha) * param2
            ).clone()
    return new_model_state_dict

