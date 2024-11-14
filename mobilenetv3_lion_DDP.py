import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import StepLR
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
from lion_pytorch import Lion
from PIL import ImageFile
import logging
from pathlib import Path

from utils.transform import transform
from run.train import ModelTrainer


class DistributedDatasetManager:
    """Dataset preparation and management class for distributed training."""

    def __init__(self, train_path: str, valid_path: str, batch_size: int = 64):
        self.batch_size = batch_size
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def prepare_datasets(self, train_ratio: float = 0.8):
        """Prepare and split datasets."""
        train_dataset = datasets.ImageFolder(root=self.train_path, transform=transform())
        valid_dataset = datasets.ImageFolder(root=self.valid_path, transform=transform())
        combined_dataset = ConcatDataset([train_dataset, valid_dataset])

        train_size = int(len(combined_dataset) * train_ratio)
        val_size = len(combined_dataset) - train_size

        return random_split(combined_dataset, [train_size, val_size])

    def create_dataloaders(self, train_dataset, val_dataset, rank, world_size):
        """Create distributed DataLoader objects."""
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True
        )

        valid_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=8,
            pin_memory=True
        )

        return train_loader, valid_loader, train_sampler, valid_sampler


class DistributedModelManager:
    """Model preparation and management class for distributed training."""

    def __init__(self, num_classes: int, rank: int):
        self.num_classes = num_classes
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")

    def prepare_model(self):
        """Prepare and modify the model for distributed training."""
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.rank])
        return model

    @staticmethod
    def prepare_training_components(model, learning_rate=1e-4):
        """Prepare training components."""
        criterion = nn.CrossEntropyLoss()
        optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        return criterion, optimizer, scheduler


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def setup_distributed(rank, world_size, master_port='12355'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_process(rank, world_size):
    """Training process for each GPU."""
    setup_distributed(rank, world_size)
    logger = logging.getLogger(f"Trainer-{rank}")

    try:
        # Initialize managers
        dataset_manager = DistributedDatasetManager(
            train_path="D:/4type_weather_driving_dataset/Training",
            valid_path="D:/4type_weather_driving_dataset/Validation"
        )
        model_manager = DistributedModelManager(num_classes=5, rank=rank)

        # Prepare datasets and dataloaders
        train_dataset, val_dataset = dataset_manager.prepare_datasets(train_ratio=0.8)
        train_loader, valid_loader, train_sampler, valid_sampler = dataset_manager.create_dataloaders(
            train_dataset, val_dataset, rank, world_size
        )

        if rank == 0:
            logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

        # Prepare model and training components
        model = model_manager.prepare_model()
        criterion, optimizer, scheduler = model_manager.prepare_training_components(model)

        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=model_manager.device,
            save_path='checkpoints',
            model_name=f'mobilenet_v3_weather_gpu{rank}'
        )

        # Start training
        if rank == 0:
            logger.info("Starting distributed training...")

        trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=50,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler
        )

        if rank == 0:
            logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred on GPU {rank}: {str(e)}", exc_info=True)
    finally:
        cleanup_distributed()


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Set master GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'  # GPU 2,3,4,5를 0,1,2,3으로 매핑
    world_size = 4  # 사용할 GPU 개수

    try:
        # Start distributed training
        mp.spawn(
            train_process,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"An error occurred in main process: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()