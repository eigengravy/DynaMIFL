import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

# from minepy import MINE
from scipy.stats import kendalltau
import math

#mine = MINE()


def load_dataset(partitioners, batch_size=64, test_size=0.1):

    fds = FederatedDataset(
        dataset="cifar100",
        partitioners={"train": partitioners},
    )

    def apply_transforms(batch):

        batch["img"] = [
            Compose(
                [
                    ToTensor(),
                    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )(img)
            for img in batch["img"]
        ]
        return batch

    testloader = DataLoader(
        fds.load_split("test").with_transform(apply_transforms), batch_size=batch_size
    )

    def get_client_loader(cid: str):
        client_dataset = fds.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(
            test_size=test_size, seed=42
        )
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]
        trainloader = DataLoader(
            trainset.with_transform(apply_transforms), batch_size=batch_size
        )
        valloader = DataLoader(
            valset.with_transform(apply_transforms), batch_size=batch_size
        )
        return trainloader, valloader

    return testloader, get_client_loader


def evaluate(model: nn.Module, test_loader: DataLoader, device) -> Tuple[float, float]:
    model.eval().to(device)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            outputs = model(images)
            test_loss += nn.functional.cross_entropy(
                outputs, labels, reduction="sum"
            ).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def calculate_mi(modelA: nn.Module, modelB: nn.Module, dataloader: DataLoader, device):
    modelA.eval()
    modelB.eval()

    modelA_outputs = []
    modelB_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch["img"].to(device), batch["fine_label"].to(device)
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
) -> nn.Module:
    model.to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()

    local_model.to(device)
    ce_loss_sum = 0
    total_loss_sum = 0
    for _ in range(epochs):
        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
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
    mi_loss_lambda: float,
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
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = nn.functional.cross_entropy(outputs, labels)
            ce_loss_sum += ce_loss.item()
            with torch.no_grad():
                local_ouputs = local_model(images)
            mi_loss = nn.functional.cross_entropy(outputs, local_ouputs)
            mi_loss = torch.clamp(mi_loss, min=min_clamp, max=max_clamp)
            mi_loss_sum += mi_loss.item()
            loss = ce_loss - mi_loss_lambda * mi_loss
            total_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
    return ce_loss_sum, mi_loss_sum, total_loss_sum

def client_mifl_update_anshul(
    model: nn.Module,
    global_model: nn.Module,
    local_model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    epochs: int,
    device,
    min_clamp: float,
    max_clamp: float,
    mi_loss_lambda: float,
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
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = nn.functional.cross_entropy(outputs, labels)
            ce_loss_sum += ce_loss.item()
            with torch.no_grad():
                local_ouputs = local_model(images)
            mi_loss = nn.functional.cross_entropy(outputs, local_ouputs)
            mi_loss = torch.clamp(mi_loss, min=min_clamp, max=max_clamp)
            mi_loss_sum += mi_loss.item()
            loss = ce_loss - calculate_lambda_anshul2(outputs, local_ouputs) * mi_loss
            total_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
    return ce_loss_sum, mi_loss_sum, total_loss_sum


def calculate_lambda_anshul(logits_g , logits_k):
    logits_g = logits_g.float()
    logits_k = logits_k.float()
    mean_logits = (logits_g + logits_k) / 2
    
    deviation_g = logits_g - mean_logits
    deviation_k = logits_k - mean_logits
    
    covariance = torch.mean(deviation_g * deviation_k)
    sum_logits = logits_g.abs() + logits_k.abs()
    cv = covariance.sum()/sum_logits.sum()
    
    lambda_val = torch.where(cv <= 0.5, cv, 1 - cv)
    
    return lambda_val

def knn_entropy(X, k=3):
    """
    Estimate the entropy of continuous random variable X using k-nearest neighbors.
    X: PyTorch tensor of shape (n_samples, n_features)
    k: Number of nearest neighbors
    Returns: Entropy estimate
    """
    n = X.shape[0]
    
    # Pairwise distances between points in X
    dist = torch.cdist(X, X, p=2)
    
    # Sort distances and get the distance to the k-th nearest neighbor
    knn_distances, _ = torch.topk(dist, k+1, largest=False)  # k+1 because the closest point is itself
    knn_distance = knn_distances[:, k]  # k-th nearest neighbor distance
    
    # Compute entropy using KNN estimate
    volume_unit_ball = (math.pi ** (X.shape[1] / 2)) / math.gamma(X.shape[1] / 2 + 1)  # Volume of d-dimensional unit ball
    entropy_estimate = - torch.mean(torch.log(knn_distance)) + torch.log(torch.tensor(volume_unit_ball)) + math.log(n) - math.log(k)
    
    return entropy_estimate

# Function to estimate mutual information using KNN
def mutual_information(X, Y, k=3):
    """
    Estimate the mutual information between continuous variables X and Y using k-nearest neighbors.
    X: PyTorch tensor of shape (n_samples, n_features)
    Y: PyTorch tensor of shape (n_samples, n_features)
    k: Number of nearest neighbors
    Returns: Mutual information estimate
    """
    H_X = knn_entropy(X, k)      # Entropy of X
    H_Y = knn_entropy(Y, k)      # Entropy of Y
    H_XY = knn_entropy(torch.cat([X, Y], dim=1), k)  # Joint entropy of X and Y
    
    # Mutual information
    MI = H_X + H_Y - H_XY
    
    return MI


def calculate_lambda_anshul2(logits_g, logits_k , k=3):
    logits_g = logits_g.float()
    logits_k = logits_k.float()

    modelA_outputs = []
    modelB_outputs = []

    # modelA_outputs.append(logits_g.detach().cpu().numpy())
    # modelB_outputs.append(logits_k.detach().cpu().numpy())
    modelA_outputs.append(logits_g.detach().cpu())
    modelB_outputs.append(logits_k.detach().cpu())

    modelA_outputs = np.concatenate(modelA_outputs, axis=0).reshape(-1)
    modelB_outputs = np.concatenate(modelB_outputs, axis=0).reshape(-1)

    std_g = torch.std(torch.tensor(modelA_outputs))
    std_k = torch.std(torch.tensor(modelB_outputs))

    rho, _ = pearsonr(modelA_outputs, modelB_outputs)

    # mine.compute_score(modelA_outputs, modelB_outputs)

    # mic = mine.mic()
    # tau , _ = kendalltau(modelA_outputs, modelB_outputs)

    mis = mutual_information(modelA_outputs.unsqueeze(1), modelB_outputs.unsqueeze(1), k=k)

    mi = -0.5 * math.log(1 - rho**2)

    summation = np.abs(modelA_outputs) + np.abs(modelB_outputs)

    cov = mis * std_g * std_k

    cv = cov.sum() / summation.sum()

    lambda_val = torch.where(cv <= 0.5, cv, 1 - cv)

    return lambda_val

