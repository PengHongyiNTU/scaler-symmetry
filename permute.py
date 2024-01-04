from typing import Callable, Tuple, Dict, Any, Literal,  Union, List
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy




def get_all_activations(
    model_builder: Callable,
    state_dict: Dict[str, Any],
    val_loader: DataLoader,
    device: Literal["cpu", "cuda"],
    activation_types: Union[type, Tuple[type, ...]],
) -> Dict[str, torch.Tensor]:
    """
    Get all activations of a model on a validation set.
    Args:
        model_builder: A callable that returns a model
        state_dict: The state dict of the model
        val_loader: The validation set data loader
        device: The device to run the model on
        activation_types: The types of activations to return
    Returns:
        A dictionary of activations
    """
    model = model_builder()
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise Exception(f"Failed to load state dict: {e}") 
    activations_by_layers = {}
    def get_layer_activations(
        module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        layer_id = id(module)
        layer_name = f"{module.__class__.__name__}_{layer_id}"
        if layer_name not in activations_by_layers:
            activations_by_layers[layer_name] = []
        activations_by_layers[layer_name].append(output.cpu().detach())
    hooks = []
    for depth, module in enumerate(model.modules()):
        if isinstance(module, activation_types):
            hook = module.register_forward_hook(get_layer_activations)
            hooks.append(hook)
    # Forward pass
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            model(inputs)
    # Remove hooks
    for hook in hooks:
        hook.remove()
    model.to("cpu")
    # Process activations
    final_activations = {}
    for depth, (layer_name, activations) in enumerate(activations_by_layers.items()):
        readable_layer_name = layer_name.split("_")[0] + f"_{depth}"
        final_activations[readable_layer_name] = torch.cat(activations, dim=0)
    return final_activations
    

def compute_matching_cost(
    z1: torch.Tensor, z2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the matching cost between two sets of activations.
    Args:
        z1: The first set of activations
        z2: The second set of activations
    Returns:
        The matching cost
    """
    assert z1.shape == z2.shape, "z1 and z2 must have the same shape"
    assert len(z1.shape) == 2, "z1 and z2 must be 2D tensors"
    z1 = z1.clone().detach().requires_grad_(False)
    z2 = z2.clone().detach().requires_grad_(False)
    z1_expanded = z1.unsqueeze(2)
    z2_expanded = z2.unsqueeze(1)
    cost_matrix = torch.norm(z1_expanded - z2_expanded, dim=0)
    return cost_matrix
    
    
def compute_cost_matrices(
    activations_1: Dict[str, torch.Tensor],
    activations_2: Dict[str, torch.Tensor]
) -> List[np.ndarray]:
    """
    Compute the cost matrices between two sets of activations.
    Args:
        activations_1: The first set of activations
        activations_2: The second set of activations
    Returns:
        A list of cost matrices
    """
    cost_matrices = []
    for layer_name in activations_1:
        z1 = activations_1[layer_name]
        z2 = activations_2[layer_name]
        cost_matrix = compute_matching_cost(z1, z2)
        cost_matrix = cost_matrix.cpu().numpy()
        cost_matrices.append(cost_matrix)
    return cost_matrices


def permutation_mapping(cost_matrix: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the permutation mapping from a cost matrix.
    Args:
        cost_matrix: The cost matrix
    Returns:
        The permutation mapping and the cost
    """
    assert len(cost_matrix.shape) == 2, "cost_matrix must be a 2D matrix"
    assert cost_matrix.shape[0] == cost_matrix.shape[1], "cost_matrix must be square"
    n = cost_matrix.shape[0]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def all_permutation_mappings(cost_matrices: List[np.ndarray],
                             verbose: bool = False
                             ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute all permutation mappings from a list of cost matrices.
    Args:
        cost_matrices: A list of cost matrices
    Returns:
        A list of permutation mappings
    """
    permutation_mappings = []
    for i, cost_matrix in enumerate(cost_matrices):
        row_ind, col_ind = permutation_mapping(cost_matrix)
        if verbose:
            cost_before = np.diagonal(cost_matrix).sum()
            cost_after = cost_matrix[row_ind, col_ind].sum()
            print(f"Layer {i}: Cost before: {cost_before}, Cost after: {cost_after}")
        permutation_mappings.append(col_ind)
    return permutation_mappings


def model_permutation(
    model: nn.Module,
    permutation_mappings: List[np.ndarray],
) -> Dict[str, torch.Tensor]:
    """
    Apply a permutation mapping to a model.
    Args:
        model: The model
        permutation_mappings: A list of permutation mappings
    Returns:
        The permuted model
    """
    linear_layers = []
    for module in model.models():
        if isinstance(module, nn.Linear):
            linear_layers.append(module)
    assert len(linear_layers) == len(permutation_mappings) + 1, (
        "Number of linear layers must be equal to the number of permutation mappings + 1"
    )
    for i in range(len(linear_layers) - 1):
        permutation = permutation_mappings[i]
        linear_layers[i].weight.data = linear_layers[i].weight.data[permutation]
        if linear_layers[i].bias is not None:
            linear_layers[i].bias.data = linear_layers[i].bias.data[permutation]
        next_layer = linear_layers[i + 1]
        next_layer.weight.data = next_layer.weight.data[:, permutation]
    new_state_dict = deepcopy(model.state_dict())
    return new_state_dict


def match_and_permute(
    model_builder: Callable,
    state_dict_1: Dict[str, Any],
    state_dict_2: Dict[str, Any],
    val_loader: DataLoader,
    device: Literal["cpu", "cuda"],
    activation_types: Union[type, Tuple[type, ...]]=(nn.ReLu, nn.Softmax),
)-> Dict[str, torch.Tensor]:
    activations_1 = get_all_activations(
        model_builder=model_builder,
        state_dict=state_dict_1,
        val_loader=val_loader,
        device=device,
        activation_types=activation_types,
    )
    activations_2 = get_all_activations(
        model_builder=model_builder,
        state_dict=state_dict_2,
        val_loader=val_loader,
        device=device,
        activation_types=activation_types,
    )
    cost_matrices = compute_cost_matrices(activations_1, activations_2)
    permutation_mappings = all_permutation_mappings(cost_matrices)
    model = model_builder()
    model.load_state_dict(state_dict_2)
    new_state_dict = model_permutation(model, permutation_mappings)
    return new_state_dict
    
  