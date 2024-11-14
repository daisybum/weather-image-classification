import torch
import torch.nn as nn
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


class DatasetManager:
    """Dataset preparation and management class."""

    def __init__(self, train_path: str, valid_path: str, batch_size: int = 64):
        self.batch_size = batch_size
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def prepare_datasets(self, train_ratio: float = 0.8):
        """Prepare and split datasets."""
        # Load initial datasets
        train_dataset = datasets.ImageFolder(root=self.train_path, transform=transform())
        valid_dataset = datasets.ImageFolder(root=self.valid_path, transform=transform())

        # Combine datasets
        combined_dataset = ConcatDataset([train_dataset, valid_dataset])

        # Calculate split sizes
        train_size = int(len(combined_dataset) * train_ratio)
        val_size = len(combined_dataset) - train_size

        # Split dataset
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        """Create DataLoader objects."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

        valid_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

        return train_loader, valid_loader


class ModelManager:
    """Model preparation and management class."""

    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device

    def prepare_model(self):
        """Prepare and modify the model."""
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        return model.to(self.device)

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


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize managers
        dataset_manager = DatasetManager(
            train_path="D:/4type_weather_driving_dataset/Training",
            valid_path="D:/4type_weather_driving_dataset/Validation"
        )
        model_manager = ModelManager(num_classes=5, device=device)

        # Prepare datasets and dataloaders
        train_dataset, val_dataset = dataset_manager.prepare_datasets(train_ratio=0.8)
        train_loader, valid_loader = dataset_manager.create_dataloaders(train_dataset, val_dataset)
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
            device=device,
            save_path='checkpoints',
            model_name='mobilenet_v3_weather'
        )

        # Start training
        logger.info("Starting training...")
        trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=50
        )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()