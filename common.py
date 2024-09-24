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
