import torch
import torch.nn as nn
import torch.nn.functional as F
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
import math

from models.simple_cnn import SimpleCNN
from workloads.cifar100 import *
from common import *

DEVICE_ARG = "cuda:1"
DEVICE = torch.device(DEVICE_ARG if torch.cuda.is_available() else "cpu")

print("Device: {DEVICE}")

from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from workloads.cifar100 import client_fedavg_update, evaluate
from common import federated_averaging, calculate_mi

import wandb


# Set the partitioner to create 10 partitions
# Example usage:
# correlation, mutual_information = calculate_mi(model, teacher, dataloader)
# print(f"Correlation coefficient (rho): {correlation}")
# print(f"Mutual Information: {mutual_information}")i
def federated_learning(
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    alpha: float,
    participation_fraction: float,

    start = datetime.now()
):
    ALPHA  = 0.5
    datasets = load_dataset(num_clients)
    # client_loaders, test_loader = load_partition_cifar10(num_clients, batch_size, alpha)
    partitioner = DirichletPartitioner(num_partitions=num_clients , partition_by="fine_label" , alpha=ALPHA)


    test_loader, get_client_loader = load_dataset(partitioner)

    global_model = SimpleCNN().to(DEVICE)
    local_models = [SimpleCNN().to(DEVICE) for _ in range(num_clients)]

    losses: List[float] = []
    accuracies: List[float] = []

    min_mi = []
    max_mi = []
    mean_mi = []

    mi_history = [[None for _ in range(num_clients)] for _ in range(num_rounds)]

    os.makedirs("results", exist_ok=True)

    for round in range(num_rounds):
        num_participating_clients = max(1, int(participation_fraction * num_clients))
        participating_clients = random.sample(
            range(num_clients), num_participating_clients
        )

        round_models = []
        round_mi = []
        for client_idx in participating_clients:
            local_model = SimpleCNN().to(DEVICE)
            teacher_model = SimpleCNN().to(DEVICE)
            teacher_model.load_state_dict(local_models[client_idx].state_dict())
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            local_model = client_fedavg_update(
                local_model,
                teacher_model,
                optimizer,
                get_client_loader(client_idx),
                local_epochs,
                DEVICE
            )
            local_models[client_idx].load_state_dict(local_model.state_dict())
            round_models.append(local_model)
            _, mi = calculate_mi(local_model, teacher_model, client_loaders[client_idx], DEVICE)
            round_mi.append(mi)
            mi_history[round][client_idx] = mi

        round_models = round_models[: int(0.8 * len(round_models))]
        global_model = federated_averaging(round_models, DEVICE)

        test_loss, accuracy = evaluate(global_model, testloader, DEVICE)
        losses.append(test_loss)
        accuracies.append(accuracy)
        print(
            f"Round {round+1}/{num_rounds}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        print(
            f"Min {min(round_mi)} Max {max(round_mi)} Mean {sum(round_mi)/len(round_mi)}"
        )
        min_mi.append(min(round_mi))
        max_mi.append(max(round_mi))
        mean_mi.append(sum(round_mi) / len(round_mi))

    name = "fedavg"
    with open(f"results/result-{hex(hash(time()))[2:]}.json", "w") as f:
        dump = {
            "name": name,
            "time": start.isoformat(),
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "alpha": alpha,
            "participation_fraction": participation_fraction,
            "losses": losses,
            "accuracies": accuracies,
            "min_mi": min_mi,
            "max_mi": max_mi,
            "mean_mi": mean_mi,
            "mi_history": mi_history,
        }
        json.dump(
            dump,
            f,
        )


if __name__ == "__main__":
    num_clients = 100
    num_rounds = 100
    local_epochs = 5
    batch_size = 32
    alpha = 0.1
    participation_fraction = 0.5

    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="fed-avg",
        # Track hyperparameters and run metadata
        config={
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "alpha": alpha,
            "participation_fraction": participation_fraction,
        },
    )

    # wandb.log({"accuracy": acc, "loss": loss})

    federated_learning(
        num_clients, num_rounds, local_epochs, batch_size, alpha, participation_fraction
    )
