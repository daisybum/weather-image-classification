import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from lion_pytorch import Lion

def create_optimizer(model: nn.Module,
                    optimizer_name: str,
                    learning_rate: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'lion':
        return Lion(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_name: str,
                    **scheduler_params) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    """
    if scheduler_name.lower() == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            **scheduler_params
        )
    elif scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **scheduler_params
        )
    elif scheduler_name.lower() == 'reduceonplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_params
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")