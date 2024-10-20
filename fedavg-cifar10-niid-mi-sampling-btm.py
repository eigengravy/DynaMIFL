import torch
from torch import optim
import wandb
from flwr_datasets.partitioner import DirichletPartitioner
from utils import client_fedavg_update, federated_averaging, evaluate, calculate_mi
from models.simple_cnn import SimpleCNN
from workloads.cifar10 import load_dataset, process_batch
import random
from tqdm import tqdm

DEVICE_ARG = "cuda:1"
DEVICE = torch.device(DEVICE_ARG if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

num_clients = 100
num_rounds = 500
local_epochs = 5
batch_size = 64
participation_fraction = 0.1
partition_alpha = 0.5
aggregation_size = participation_fraction * num_clients

wandb.login()

wandb.init(
    project=f"x-experiment-fedavg-cifar10-niid-mi-sampling-btm",
    config={
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "partition_alpha": partition_alpha,
        "participation_fraction": participation_fraction,
    },
)

partitioner = DirichletPartitioner(
    num_partitions=num_clients, partition_by="label", alpha=partition_alpha
)


test_loader, get_client_loader = load_dataset(partitioner)

global_model = SimpleCNN(num_classes=10).to(DEVICE)
local_models = [SimpleCNN(num_classes=10).to(DEVICE) for _ in range(num_clients)]

local_mis = [1] * num_clients

for round in tqdm(range(num_rounds)):
    if round < 10:
        num_participating_clients = num_clients
        participating_clients = random.sample(range(num_clients), num_participating_clients)
    else:
        num_participating_clients = max(1, int(participation_fraction * num_clients))
        participating_clients = random.choices(population=range(num_clients), k=num_participating_clients, weights=local_mis)

    round_models = []
    for client_idx in participating_clients:
        trainloader, valloader = get_client_loader(client_idx)
        model = SimpleCNN(num_classes=10)
        optimizer = optim.SGD(model.parameters(), lr=0.1*((0.99)**round))
        ce_loss_sum, total_loss_sum = client_fedavg_update(
            model,
            global_model,
            local_models[client_idx],
            trainloader,
            optimizer,
            local_epochs,
            DEVICE,
            process_batch
        )
        round_models.append(model)

        test_loss, accuracy = evaluate(model, valloader, DEVICE, process_batch)
        mi = calculate_mi(model, local_models[client_idx], valloader, DEVICE, process_batch)
        if round < 10:
            local_mis[client_idx] += mi
        local_models[client_idx].load_state_dict(model.state_dict())

        wandb.log(
            {
                str(client_idx): {
                    "ce_loss_sum": ce_loss_sum,
                    "total_loss_sum": total_loss_sum,
                    "test_loss": test_loss,
                    "accuracy": accuracy,
                    "mi": mi,
                }
            },
            commit=False,
        )

    round_models = round_models[: int(aggregation_size)]
    federated_averaging(global_model, round_models, DEVICE)
    test_loss, accuracy = evaluate(global_model, test_loader, DEVICE, process_batch)
    wandb.log(
        {
            "global_loss": test_loss,
            "global_accuracy": accuracy,
            "aggregation_size": aggregation_size,
        }
    )
