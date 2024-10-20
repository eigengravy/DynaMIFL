from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter


def load_dataset(partitioners, batch_size=64, test_size=0.1):

    fds = FederatedDataset(
        dataset="cifar10",
        partitioners={"train": partitioners},
    )

    def apply_transforms(batch):

        batch["img"] = [
            Compose(
                [
                    RandomHorizontalFlip(),
                    RandomRotation(10),
                    RandomAffine(0, shear=10, scale=(0.8,1.2)),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )(img)
            for img in batch["img"]
        ]
        return batch

    def apply_test_transforms(batch):

        batch["img"] = [
            Compose(
                [
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )(img)
            for img in batch["img"]
        ]
        return batch

    testloader = DataLoader(
        fds.load_split("test").with_transform(apply_test_transforms), batch_size=batch_size
    )

    def get_client_loader(cid: str):
        client_dataset = fds.load_partition(int(cid), "train")
        # client_dataset_splits = client_dataset.train_test_split(
        #     test_size=test_size, seed=42
        # )
        # trainset = client_dataset_splits["train"]
        # valset = client_dataset_splits["test"]
        trainloader = DataLoader(
            client_dataset.with_transform(apply_transforms), batch_size=batch_size
        )
        valloader = DataLoader(
            client_dataset.with_transform(apply_test_transforms), batch_size=batch_size
        )
        return trainloader, valloader

    return testloader, get_client_loader

def process_batch(batch):
    return batch["img"], batch["label"]


