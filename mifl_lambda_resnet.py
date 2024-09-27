import torch
from torch import optim
import wandb
from flwr_datasets.partitioner import DirichletPartitioner
from common import federated_averaging
from models.simple_cnn import SimpleCNN
from models.resnet50 import ResNet50
from workloads.cifar100 import (
    calculate_mi,
    client_fedavg_update,
    client_mifl_update,
    client_mifl_update_anshul,
    evaluate,
    load_dataset,
)
import random
import numpy as np
from tqdm import tqdm


DEVICE_ARG = "cuda:0"
DEVICE = torch.device(DEVICE_ARG if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

num_clients = 100
num_rounds = 100
local_epochs = 5
batch_size = 32
partition_alpha = 0.1
participation_fraction = 0.3
mifl_lambda = 0.4
mifl_clamp = 5
mifl_critical_value = 0.025
aggregation_size = 0.8 * participation_fraction * num_clients
net = ResNet50()

wandb.login()

wandb.init(
    project="mifl-lambda-resnet-base",
    config={
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "parition_alpha": partition_alpha,
        "mifl_lambda": mifl_lambda,
        "mifl_clamp": mifl_clamp,
        "participation_fraction": participation_fraction, },)

partitioner = DirichletPartitioner(
    num_partitions=num_clients, partition_by="fine_label", alpha=partition_alpha
)

test_loader, get_client_loader = load_dataset(partitioner)

global_model = ResNet50().to(DEVICE)
local_models = [ResNet50().to(DEVICE) for _ in range(num_clients)]


for round in tqdm(range(num_rounds)):
    num_participating_clients = max(1, int(participation_fraction * num_clients))
    participating_clients = random.sample(range(num_clients), num_participating_clients)

#    if round % 10 == 0 and round > 0:
#        mifl_critical_value -= 0.025

    round_models = []
    round_mis = []
    for client_idx in participating_clients:
        trainloader, valloader = get_client_loader(client_idx)
        model = ResNet50().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        if round == 0:
            ce_loss_sum, total_loss_sum = client_fedavg_update(
                model,
                global_model,
                local_models[client_idx],
                trainloader,
                optimizer,
                local_epochs,
                DEVICE
            )
            mi_loss_sum = 0
        else:
            ce_loss_sum, mi_loss_sum, total_loss_sum = client_mifl_update_anshul(
                model,
                global_model,
                local_models[client_idx],
                trainloader,
                optimizer,
                local_epochs,
                DEVICE,
                -mifl_clamp,
                mifl_clamp,
                mifl_lambda,
            )
        round_models.append(model)
        test_loss, accuracy = evaluate(model, valloader, DEVICE)
        mi = calculate_mi(model, local_models[client_idx], trainloader, DEVICE)
        local_models[client_idx].load_state_dict(model.state_dict())
        round_mis.append(mi)

        wandb.log(
            {
                str(client_idx): {
                    "ce_loss_sum": ce_loss_sum,
                    "mi_loss_sum": mi_loss_sum,
                    "total_loss_sum": total_loss_sum,
                    "test_loss": test_loss,
                    "accuracy": accuracy,
                    "mi": mi,
                }
            },
            commit=False,
        )

    # print(f"Round {round}")
    # print(f"{len(round_models)} {round_mis}")
    lower_bound_mi = np.nanpercentile(round_mis, mifl_critical_value * 100)
    upper_bound_mi = np.nanpercentile(round_mis, (1 - mifl_critical_value) * 100)
    merged = [
        (_mi, _model)
        for _mi, _model in zip(round_mis, round_models)
        if lower_bound_mi <= _mi <= upper_bound_mi
    ]
    merged.sort()
    round_models = [_model for (_mi, _model) in merged[: int(aggregation_size)]]
    federated_averaging(global_model, round_models, DEVICE)
    test_loss, accuracy = evaluate(global_model, test_loader, DEVICE)
    wandb.log(
        {
            "global_loss": test_loss,
            "lower_bound_mi": lower_bound_mi,
            "upper_bound_mi": upper_bound_mi,
            "global_accuracy": accuracy,
            "aggregation_size": len(round_models),
            "mifl_critical_value": mifl_critical_value,
        }
    )

