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
# DEVICE = torch.device(DEVICE_ARG if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")

print(f"Device: {DEVICE}")

from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from workloads.cifar100 import client_fedavg_update, evaluate
from common import federated_averaging, calculate_mi

import wandb


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
    partitioner = DirichletPartitioner(num_partitions=num_clients , partition_by="fine_label" , alpha=ALPHA)

    test_loader, get_client_loader = load_dataset(partitioner)

    global_model = SimpleCNN().to(DEVICE)
    local_models = [SimpleCNN().to(DEVICE) for _ in range(num_clients)]
    local_runs = [wandb.init(entity=str(i), project="fed-avg") for i in range(num_clients)]
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
            trainloader , _ = get_client_loader(client_idx)
            local_model, ce_loss_sum, total_loss_sum = client_fedavg_update(global_model, local_models[client_idx], trainloader, local_epochs, DEVICE)
            round_models.append(local_model)
            mi  = calculate_mi(local_model, local_models[client_idx], trainloader, DEVICE)
            #wandb.log({"mi": mi , "client_idx": client_idx})
            local_models[client_idx].load_state_dict(local_model.state_dict())
            round_mi.append(mi)
            mi_history[round][client_idx] = mi

            wandb.log({str(client_idx): {"ce_loss_sum": ce_loss_sum, "total_loss_sum": total_loss_sum, "mi": mi}}, commit=False)

        round_models = round_models[: int(0.8 * len(round_models))]
        global_model = federated_averaging(round_models, DEVICE)

        test_loss, accuracy = evaluate(global_model, test_loader, DEVICE)
        wandb.log({"test_loss": test_loss, "accuracy": accuracy})
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

    # name = "fedavg"
    # with open(f"results/result-{hex(hash(time()))[2:]}.json", "w") as f:
    #     dump = {
    #         "name": name,
    #         "time": start.isoformat(),
    #         "num_clients": num_clients,
    #         "num_rounds": num_rounds,
    #         "local_epochs": local_epochs,
    #         "batch_size": batch_size,
    #         "alpha": alpha,
    #         "participation_fraction": participation_fraction,
    #         "losses": losses,
    #         "accuracies": accuracies,
    #         "min_mi": min_mi,
    #         "max_mi": max_mi,
    #         "mean_mi": mean_mi,
    #         "mi_history": mi_history,
    #     }
    #     json.dump(
    #         dump,
    #         f,
    #     )
    global_run.log({"losses":losses, "accuracies":accuracies, "min_mi":min_mi, "max_mi":max_mi, "mean_mi":mean_mi, "mi_history":mi_history})


if __name__ == "__main__":
    num_clients = 100
    num_rounds = 100
    local_epochs = 5
    batch_size = 32
    alpha = 0.1
    participation_fraction = 0.5

    wandb.login()

    global_run = wandb.init(
        project="fed-avg",
        config={
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "alpha": alpha,
            "participation_fraction": participation_fraction,
        },
    )

    federated_learning(
        num_clients, num_rounds, local_epochs, batch_size, alpha, participation_fraction
    )
