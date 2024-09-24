import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor


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
