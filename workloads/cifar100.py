import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from flwr_datasets import FederatedDataset

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
from models.simple_cnn import SimpleCNN
import torch.optim as optim
from typing import List, Tuple


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

def client_fedavg_update(
    global_model: nn.Module,
    local_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device
) -> nn.Module:
    model = SimpleCNN().to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    
    local_model.to(device)
    
    for _ in range(epochs):
        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # TODO: Log
            ce_loss = nn.functional.cross_entropy(outputs, labels)       
            # TODO: Log    
            loss = ce_loss
            loss.backward()
            optimizer.step()
    return model


def evaluate(model: nn.Module, test_loader: DataLoader,device) -> Tuple[float, float]:
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

def client_mifl_update(
    global_model: nn.Module,
    local_model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int,
    alpha: float,
    device
) -> nn.Module:
    model = SimpleCNN().to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()
    
    local_model.to(device)
    
    for _ in range(epochs):
        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # TODO: Log
            ce_loss = nn.functional.cross_entropy(outputs, labels)       
            # TODO: Log
            with torch.no_grad():
                local_ouputs = local_model(outputs, labels)
            # TODO: mi loss
            mi_loss = nn.functional.cross_entropy(outputs, local_ouputs)
            loss = ce_loss - alpha * mi_loss
            loss.backward()
            optimizer.step()
    return model