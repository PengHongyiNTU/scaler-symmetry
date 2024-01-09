import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Literal, Tuple, List
import torchvision.transforms.v2 as transforms



def prepare_sns():
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.set(font='DejaVu Serif', font_scale=1)
    


def build_mlp_model(hidden_dimension: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Linear(28 * 28, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, 10),
    )


def prepare_dataset(
    dataset_name: Literal["MNIST", "CIFAR10"], model_type: Literal["MLP", "CNN"] = "MLP"
) -> Tuple[Dataset, Dataset]:
    if dataset_name == "MNIST":
        train_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.RandomRotation((-7, 7)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(
                    lambda x: x.flatten()
                    if model_type == "MLP"
                    else x.view(-1, 1, 28, 28)
                ),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(
                    lambda x: x.flatten()
                    if model_type == "MLP"
                    else x.view(-1, 1, 28, 28)
                ),
            ]
        )
        trainset = datasets.MNIST(
            root="data", train=True, download=True, transform=train_transforms
        )
        testset = datasets.MNIST(
            root="data", train=False, download=True, transform=test_transforms
        )
    elif dataset_name == "CIFAR10":
        train_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Lambda(lambda x: x.flatten() if model_type == "MLP" else x),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Lambda(lambda x: x.flatten() if model_type == "MLP" else x),
            ]
        )
        trainset = datasets.CIFAR10(
            root="data", train=True, download=True, transform=train_transforms
        )
        testset = datasets.CIFAR10(
            root="data", train=False, download=True, transform=test_transforms
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return trainset, testset

def parameter_wise_difference(
    models_state_dict: List[dict], 
    type: Literal["l1", "l2", "cos"] = "l1"
):
    diffs = {}
    base_model_dict = models_state_dict[0]
    # no gradient for this
    with torch.no_grad():
        for i, model_dict in enumerate(models_state_dict[1:]):
            layer_wise_diffs = []

            # iterate through each linear layer in the base model
            for key in base_model_dict.keys():
                if 'weight' in key or 'bias' in key:
                    base_param = base_model_dict[key]
                    param = model_dict[key]
                    if type == "l1":
                        diff = torch.norm(base_param - param, p=1)
                    elif type == "l2":
                        diff = torch.norm(base_param - param, p=2)
                    elif type == "cos":
                        cos_sim = torch.nn.functional.cosine_similarity(
                            base_param.view(1, -1), param.view(1, -1), dim=1
                        )
                        diff = cos_sim.mean()
                    else:
                        raise ValueError(f"Unknown type: {type}")
                    layer_wise_diffs.append(diff.item())
            diffs[i + 1] = layer_wise_diffs
    return diffs