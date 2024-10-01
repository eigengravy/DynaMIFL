import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from scipy.stats import pearsonr, kendalltau
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

def process_batch(batch):
    return batch["img"], batch["fine_label"]


