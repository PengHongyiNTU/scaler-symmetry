import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Literal, Tuple


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
                transforms.RandomRotation(7),
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

