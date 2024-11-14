import os
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional
import logging


@dataclass
class TrainingMetrics:
    """Class for storing training metrics."""
    loss: float
    accuracy: float

    def __str__(self) -> str:
        return f'Loss: {self.loss:.4f}, Accuracy: {self.accuracy:.2f}%'

    def reduce(self):
        """Reduce metrics across all processes."""
        metrics = torch.tensor([self.loss, self.accuracy]).to('cuda')
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        metrics /= world_size
        return TrainingMetrics(loss=metrics[0].item(), accuracy=metrics[1].item())


class DistributedModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: torch.device,
                 save_path: str = 'model',
                 model_name: str = 'model'):
        """Initialize ModelTrainer with model and training components."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.model_name = model_name
        self.scaler = GradScaler()
        self.logger = self._setup_logger()
        self.rank = dist.get_rank()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _compute_metrics(self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> TrainingMetrics:
        """Compute loss and accuracy metrics."""
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return TrainingMetrics(loss=loss.item(), accuracy=accuracy)

    def train_epoch(self, loader: torch.utils.data.DataLoader,
                    sampler: Optional[torch.utils.data.DistributedSampler] = None) -> TrainingMetrics:
        """Train the model for one epoch."""
        self.model.train()
        running_metrics = TrainingMetrics(loss=0.0, accuracy=0.0)
        total_batches = len(loader)

        if sampler:
            sampler.set_epoch(self.current_epoch)

        with tqdm(loader, desc=f"Training (GPU {self.rank})", leave=False, disable=self.rank != 0) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_metrics = self._compute_metrics(outputs, labels, loss)
                running_metrics.loss += batch_metrics.loss
                running_metrics.accuracy += batch_metrics.accuracy

                if self.rank == 0:
                    pbar.set_postfix({'loss': f'{batch_metrics.loss:.4f}',
                                      'accuracy': f'{batch_metrics.accuracy:.2f}%'})

        metrics = TrainingMetrics(
            loss=running_metrics.loss / total_batches,
            accuracy=running_metrics.accuracy / total_batches
        )

        # Synchronize metrics across all processes
        return metrics.reduce()

    def validate_epoch(self, loader: torch.utils.data.DataLoader,
                       sampler: Optional[torch.utils.data.DistributedSampler] = None) -> TrainingMetrics:
        """Validate the model."""
        self.model.eval()
        running_metrics = TrainingMetrics(loss=0.0, accuracy=0.0)
        total_batches = len(loader)

        if sampler:
            sampler.set_epoch(self.current_epoch)

        with torch.no_grad():
            with tqdm(loader, desc=f"Validation (GPU {self.rank})", leave=False, disable=self.rank != 0) as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    batch_metrics = self._compute_metrics(outputs, labels, loss)
                    running_metrics.loss += batch_metrics.loss
                    running_metrics.accuracy += batch_metrics.accuracy

                    if self.rank == 0:
                        pbar.set_postfix({'loss': f'{batch_metrics.loss:.4f}',
                                          'accuracy': f'{batch_metrics.accuracy:.2f}%'})

        metrics = TrainingMetrics(
            loss=running_metrics.loss / total_batches,
            accuracy=running_metrics.accuracy / total_batches
        )

        # Synchronize metrics across all processes
        return metrics.reduce()

    def save_checkpoint(self, epoch: int) -> None:
        """Save the model checkpoint."""
        if self.rank == 0:  # Only save on master process
            try:
                os.makedirs(self.save_path, exist_ok=True)
                model_path = os.path.join(self.save_path, f'{self.model_name}_epoch_{epoch + 1}.pth')
                state_path = os.path.join(self.save_path, f'{self.model_name}_state_epoch_{epoch + 1}.pth')

                torch.save(self.model.module.state_dict(), model_path)  # Save the inner model
                torch.save({
                    'epoch': epoch + 1,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, state_path)

                self.logger.info(f"Checkpoint saved successfully at epoch {epoch + 1}")
            except Exception as e:
                self.logger.error(f"Error saving checkpoint: {e}")

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader,
              num_epochs: int = 10,
              train_sampler: Optional[torch.utils.data.DistributedSampler] = None,
              valid_sampler: Optional[torch.utils.data.DistributedSampler] = None) -> None:
        """Train and validate the model."""
        best_valid_loss = float('inf')

        for epoch in range(num_epochs):
            self.current_epoch = epoch  # Store current epoch for samplers
            train_metrics = self.train_epoch(train_loader, train_sampler)
            valid_metrics = self.validate_epoch(valid_loader, valid_sampler)

            if self.rank == 0:  # Only log on master process
                self.logger.info(f'\nEpoch {epoch + 1}/{num_epochs}')
                self.logger.info(f'Training - {train_metrics}')
                self.logger.info(f'Validation - {valid_metrics}')

                if valid_metrics.loss < best_valid_loss:
                    best_valid_loss = valid_metrics.loss
                    self.save_checkpoint(epoch)

            self.scheduler.step()
            dist.barrier()  # Synchronize processes after each epoch