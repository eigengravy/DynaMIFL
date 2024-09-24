import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Tuple
import random
from time import time
import os
import json
from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float
) -> List[np.ndarray]:

    client_idcs = [[] for _ in range(num_clients)]

    while any(len(idcs) == 0 for idcs in client_idcs):

        n_classes = len(np.unique(labels))
        label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)
        class_idcs = [np.argwhere(labels == y).flatten() for y in range(n_classes)]
        client_idcs = [[] for _ in range(num_clients)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(
                np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
            ):
                client_idcs[i] += [idcs]
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


# Load and partition CIFAR-10 dataset
def load_partition_cifar10(
    num_clients: int, batch_size: int, alpha: float
) -> Tuple[List[DataLoader], DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    labels = np.array(trainset.targets)
    client_idcs = dirichlet_partition(labels, num_clients, alpha)
    client_loaders = [
        DataLoader(Subset(trainset, idcs), batch_size=batch_size, shuffle=True)
        for idcs in client_idcs
    ]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader
