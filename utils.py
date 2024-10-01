
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from scipy.stats import pearsonr, kendalltau
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

from typing import List

import torch
import torch.nn as nn


def federated_averaging(
    global_model: nn.Module, models: List[nn.Module], device
) -> nn.Module:
    global_model.to(device)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_model.state_dict()[k].float() for client_model in models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate(model: nn.Module, test_loader: DataLoader, device, process_batch) -> Tuple[float, float]:
    model.eval().to(device)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = process_batch(batch)
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



def calculate_mi(modelA: nn.Module, modelB: nn.Module, dataloader: DataLoader, device, process_batch) -> float:
    modelA.eval()
    modelB.eval()

    modelA_outputs = []
    modelB_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            images, _ = process_batch(batch)
            images = images.to(device)            
            modelA_outputs.append(modelA(images).cpu().numpy())
            modelB_outputs.append(modelB(images).cpu().numpy())

    modelA_outputs = np.concatenate(modelA_outputs, axis=0).reshape(-1)
    modelB_outputs = np.concatenate(modelB_outputs, axis=0).reshape(-1)

    rho, _ = pearsonr(modelA_outputs, modelB_outputs)
    mi = -0.5 * math.log(1 - rho**2)

    return mi


def client_fedavg_update(
    model: nn.Module,
    global_model: nn.Module,
    local_model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    epochs: int,
    device,
    process_batch,
) -> nn.Module:
    model.to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()

    local_model.to(device)
    ce_loss_sum = 0
    total_loss_sum = 0
    for _ in range(epochs):
        for batch in train_loader:
            images, labels = process_batch(batch)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = nn.functional.cross_entropy(outputs, labels)
            ce_loss_sum += ce_loss.item()
            loss = ce_loss
            total_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
    return ce_loss_sum, total_loss_sum


def client_mifl_update(
    model: nn.Module,
    global_model: nn.Module,
    local_model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    epochs: int,
    device,
    min_clamp: float,
    max_clamp: float,
    process_batch,
    mi_loss_lambda: float = None,
) -> nn.Module:
    model.to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()
    local_model.to(device)
    ce_loss_sum = 0
    mi_loss_sum = 0
    total_loss_sum = 0
    for _ in range(epochs):
        for batch in train_loader:
            images, labels = process_batch(batch)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = nn.functional.cross_entropy(outputs, labels)
            ce_loss_sum += ce_loss.item()
            with torch.no_grad():
                local_outputs = local_model(images)
            mi_loss = nn.functional.cross_entropy(outputs, local_outputs)
            mi_loss = torch.clamp(mi_loss, min=min_clamp, max=max_clamp)
            mi_loss_sum += mi_loss.item()
            lambda_val = calculate_lambda(outputs, local_outputs)
            # Override for testing
            if mi_loss_lambda:
                lambda_val = mi_loss_lambda
            loss = ce_loss - lambda_val * mi_loss
            total_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
    return ce_loss_sum, mi_loss_sum, total_loss_sum

def calculate_lambda(logits_g, logits_k):
    logits_g = logits_g.float()
    logits_k = logits_k.float()

    modelA_outputs = []
    modelB_outputs = []

    modelA_outputs.append(logits_g.detach().cpu().numpy())
    modelB_outputs.append(logits_k.detach().cpu().numpy())
 

    modelA_outputs = np.concatenate(modelA_outputs, axis=0).reshape(-1)
    modelB_outputs = np.concatenate(modelB_outputs, axis=0).reshape(-1)

    std_g = torch.std(torch.tensor(modelA_outputs))
    std_k = torch.std(torch.tensor(modelB_outputs))

    tau , _ = kendalltau(modelA_outputs, modelB_outputs)
    
    summation = np.abs(modelA_outputs) + np.abs(modelB_outputs)

    cov = tau * std_g * std_k

    cv = cov.sum() / summation.sum()

    lambda_val = torch.where(cv <= 0.5, cv, 1 - cv)

    return lambda_val
