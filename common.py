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
from workloads.cifar100 import load_dataset

def federated_averaging(models: List[nn.Module], device) -> nn.Module:
    global_model = SimpleCNN().to(device)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_model.state_dict()[k].float() for client_model in models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


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
