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
from datasets.cifar100 import load_dataset

DEVICE = "cuda:1"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

print("Device: {device}")


def client_update(
    model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, epochs: int, teacher: nn.Module
) -> nn.Module:
    model.train()
    model.to(device)
    teacher.to(device)
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(images)
            logits = model(images)
            ce_loss = nn.functional.cross_entropy(logits, labels)
            #mi_loss = NCL()(logits, teacher_logits, labels)
            #mi_loss = nn.functional.cross_entropy(logits, teacher_logits, reduce = False)
            #_lambda = calculate_lambda(logits, teacher_logits, labels)
#            print(_lambda)
            loss = ce_loss
#            print(ce_loss, mi_loss, loss)
#            print(loss)
            if torch.isnan(loss):
                os._exit(0)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    model.to(device)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += nn.functional.cross_entropy(
                outputs, labels, reduction="sum"
            ).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy




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
):
    start = datetime.now()

    client_loaders, test_loader = load_partition_cifar10(num_clients, batch_size, alpha)

    global_model = SimpleCNN().to(device)
    local_models = [SimpleCNN().to(device) for _ in range(num_clients)]

    losses : List[float] = []
    accuracies : List[float] = []

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
            local_model = SimpleCNN().to(device)
            teacher_model = SimpleCNN().to(device)
            teacher_model.load_state_dict(local_models[client_idx].state_dict())
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            local_model = client_update(
                local_model, optimizer, client_loaders[client_idx], local_epochs, teacher_model
            )
            local_models[client_idx].load_state_dict(local_model.state_dict())
            round_models.append(local_model)
            _, mi = calculate_mi(local_model, teacher_model, client_loaders[client_idx])
            round_mi.append(mi)
            mi_history[round][client_idx] = mi
        
        round_models = round_models[:int(0.8*len(round_models))]
        global_model = federated_averaging(round_models)

        test_loss, accuracy = evaluate(global_model, test_loader)
        losses.append(test_loss)
        accuracies.append(accuracy)
        print(
            f"Round {round+1}/{num_rounds}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        print(f"Min {min(round_mi)} Max {max(round_mi)} Mean {sum(round_mi)/len(round_mi)}")
        min_mi.append(min(round_mi))
        max_mi.append(max(round_mi))
        mean_mi.append(sum(round_mi)/len(round_mi))

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
            "mi_history": mi_history
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

    federated_learning(
        num_clients, num_rounds, local_epochs, batch_size, alpha, participation_fraction
    )
