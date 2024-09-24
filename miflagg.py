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

from common2 import SimpleCNN, load_partition_cifar10


# _lambda = 0.3

# Check if CUDA is available and set the device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print(f"Using device: {device}")

def calculate_lambda(logits_g, logits_k, targets):
    # Ensure inputs are float tensors
    logits_g = logits_g.float()
    logits_k = logits_k.float()
    
    # Calculate the variance between logits
    mean_logits = (logits_g + logits_k) / 2
    variance = ((logits_g - mean_logits)**2 + (logits_k - mean_logits)**2) / 2
    
    # Calculate the sum of logits
    sum_logits = logits_g.abs() + logits_k.abs()
    
    # Compute cv as the sum of variance divided by sum of logits
    cv = variance.sum(dim=1) / (sum_logits.sum(dim=1) + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Calculate lambda
    lambda_val = torch.where(cv <= 0.5, cv, 1 - cv)
    
    return lambda_val

def calculate_lambda_anshul(logits_g , logits_k , targets):
    logits_g = logits_g.float()
    logits_k = logits_k.float()

    # Step 1: Calculate the mean of logits_g and logits_k
    mean_logits = (logits_g + logits_k) / 2
    
    # Step 2: Subtract the mean from logits_g and logits_k
    deviation_g = logits_g - mean_logits
    deviation_k = logits_k - mean_logits
    
    # Step 3: Calculate the covariance (element-wise product of deviations)
    covariance = torch.mean(deviation_g * deviation_k)

    # Calculate the sum of logits
    sum_logits = logits_g.abs() + logits_k.abs()

    cv = covariance.sum()/sum_logits.sum()

    # Calculate lambda
    lambda_val = torch.where(cv <= 0.5, cv, 1 - cv)
    
    return lambda_val

def calculate_lambda2(logits_g, logits_k, targets):
    # Convert logits to probabilities
    probs_g = F.softmax(logits_g, dim=1)
    probs_k = F.softmax(logits_k, dim=1)
    
    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=logits_g.size(1)).float()
    
    # Calculate lg and lk
    lg = probs_g - targets_one_hot
    lk = probs_k - targets_one_hot
    
    # Calculate sigma (minimum of absolute values)
    sigma = torch.min(torch.abs(lg), torch.abs(lk))
    
    # Calculate Sigma (sum of absolute values)
    Sigma = torch.abs(lg) + torch.abs(lk)
    
    # Calculate cv for each sample in the batch
    cv = sigma.sum(dim=1) / Sigma.sum(dim=1)
    
    # Calculate lambda
    lambda_val = torch.where(cv <= 0.5, cv, 1 - cv)
    
    return lambda_val

import sys

def client_update(
    mini_val, max_val, model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, epochs: int, teacher: nn.Module
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
            mi_loss = nn.functional.cross_entropy(logits, teacher_logits)
            mi_loss = torch.clamp(mi_loss, min=-mini_val, max=max_val)
            lambda_val = calculate_lambda_anshul(logits, teacher_logits, labels)
            print(lambda_val)
            # _lambda_ = calculate_lambda(logits, teacher_logits, labels).mean()
            # print(_lambda_)
#            print(_lambda)
            loss = ce_loss - lambda_val * mi_loss
#            loss = loss.mean()
            # loss = ce_loss
            print(ce_loss, mi_loss, loss)
#            print(loss)
            if torch.isnan(loss):
                os._exit(0)
            loss.backward()
            optimizer.step()
    return model

class NCL(nn.Module):
    def __init__(self):
        super(NCL, self).__init__()
        self.lambda_param = 0.5

    def forward(self, student_outputs, teacher_outputs, targets):
        student_probs = F.log_softmax(student_outputs, dim=1)
        teacher_probs = F.log_softmax(teacher_outputs, dim=1)

        batch_size = student_probs.size(0)
        num_classes = student_probs.size(1)

        correlation_penalty = 0
        for i in range(batch_size):
            for c in range(num_classes):
                F_g = student_probs[i, c]
                F_k = teacher_probs[i, c]
                y = 1 if targets[i] == c else 0

                term1 = (1 - self.lambda_param) * (F_g - y) / (F_g * (1 - F_g) + 1e-12)
                term2 = self.lambda_param * (F_k - y) / (F_g * (1 - F_g) + 1e-12)
                print(term1, term2)
                correlation_penalty += term1 + term2

        correlation_penalty /= (batch_size )# * num_classes)
        total_loss = correlation_penalty

        return total_loss

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

def federated_averaging(models: List[nn.Module]) -> nn.Module:
    global_model = SimpleCNN().to(device)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_model.state_dict()[k].float() for client_model in models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
import math

def calculate_mi(model: nn.Module, teacher: nn.Module, dataloader: DataLoader):
    model.eval()
    teacher.eval()
    
    all_model_outputs = []
    all_teacher_outputs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]  # Assuming the first element of the batch is the input
            inputs = inputs.to(device) 
            model_output = model(inputs).cpu().numpy()
            teacher_output = teacher(inputs).cpu().numpy()
            
            all_model_outputs.append(model_output)
            all_teacher_outputs.append(teacher_output)
    
    # Concatenate all outputs
    all_model_outputs = np.concatenate(all_model_outputs, axis=0)
    all_teacher_outputs = np.concatenate(all_teacher_outputs, axis=0)
    
    # Flatten the outputs if they're multi-dimensional
    all_model_outputs = all_model_outputs.reshape(-1)
    all_teacher_outputs = all_teacher_outputs.reshape(-1)
    
    # Calculate Pearson correlation coefficient
    rho, _ = pearsonr(all_model_outputs, all_teacher_outputs)
    
    # Calculate Mutual Information
    mi = -0.5 * math.log(1 - rho**2)
    
    return rho, mi

# Example usage:
# correlation, mutual_information = calculate_mi(model, teacher, dataloader)
# print(f"Correlation coefficient (rho): {correlation}")
# print(f"Mutual Information: {mutual_information}")i
def federated_learning(
    mini_val: float,
    max_val: float,
    # lambda_val: float,
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

    os.makedirs(f"results/clamped-{max_val}", exist_ok=True)

    for round in range(num_rounds):
        num_participating_clients = max(1, int(participation_fraction * num_clients))
        participating_clients = random.sample(
            range(num_clients), num_participating_clients
        )
        while not participating_clients:
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
                mini_val, max_val, local_model, optimizer, client_loaders[client_idx], local_epochs, teacher_model
            )
            local_models[client_idx].load_state_dict(local_model.state_dict())
            round_models.append(local_model)
            _, mi = calculate_mi(local_model, teacher_model, client_loaders[client_idx])
            round_mi.append(mi)
        
        merged = [(_mi, _model) for _mi, _model in zip(round_mi, round_models)]
        merged.sort()
        merged = merged[:int(0.8*len(merged))]
        global_model = federated_averaging([_m for _, _m in merged])

        test_loss, accuracy = evaluate(global_model, test_loader)
        losses.append(test_loss)
        accuracies.append(accuracy)
        print(
            f"Round {round+1}/{num_rounds}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        print(f"Min {min(round_mi)} Max {max(round_mi)} Mean {sum(round_mi)/len(round_mi)}")


    name = "mifl_clamped"
    with open(f"results/clamped-{max_val}.json", "w") as f:
        dump = {
            "name": name,
            "time": start.isoformat(),
            "num_clients": num_clients, "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "alpha": alpha,
            "participation_fraction": participation_fraction,
            "losses": losses,
            "accuracies": accuracies,
            # "lambda": lambda_val,
            "max_clamp": max_val,
            "min_clamp": mini_val,
        }
        json.dump(
            dump,
            f,
        )

for max_val in np.arange(10, 110, 10):
    max_val = int(max_val)
    mini_val = max_val * -1
    # for lambda_val in np.arange(0.1, 0.6, 0.1):
    #     lambda_val = float(lambda_val)
    if __name__ == "__main__":
        num_clients = 100
        num_rounds = 100
        local_epochs = 5
        batch_size = 32
        alpha = 0.1
        participation_fraction = 0.1

        # print(lambda_val)
        print(mini_val)
        print(max_val)

        federated_learning(
            mini_val, max_val, num_clients, num_rounds, local_epochs, batch_size, alpha, participation_fraction
        )
            
            
