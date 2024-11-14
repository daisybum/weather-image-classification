import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms
from typing import Tuple, Optional
from pathlib import Path


class DistributedDatasetManager:
    """Dataset management for distributed training."""

    def __init__(self,
                 train_path: str,
                 valid_path: str,
                 batch_size: int = 64,
                 num_workers: int = 8):
        """Initialize dataset manager."""
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self._get_transforms()

    def _get_transforms(self) -> transforms.Compose:
        """Get data transformations."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def prepare_datasets(self, train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        train_dataset = datasets.ImageFolder(
            root=self.train_path,
            transform=self.transform
        )
        valid_dataset = datasets.ImageFolder(
            root=self.valid_path,
            transform=self.transform
        )
        return train_dataset, valid_dataset

    def create_dataloaders(self,
                           train_dataset: Dataset,
                           val_dataset: Dataset,
                           rank: int,
                           world_size: int) -> Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
        """Create data loaders for distributed training."""
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        valid_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

        valid_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, valid_loader, train_sampler, valid_sampler