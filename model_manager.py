import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional


class DistributedModelManager:
    """Model management for distributed training."""

    def __init__(self,
                 num_classes: int,
                 rank: int,
                 sync_bn: bool = True):
        """Initialize model manager."""
        self.num_classes = num_classes
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.sync_bn = sync_bn

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for distributed training."""
        # Convert BatchNorm to SyncBatchNorm if requested
        if self.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Move model to device
        model = model.to(self.device)

        # Wrap model with DDP
        model = DDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False
        )

        return model

    def load_checkpoint(self,
                        model: nn.Module,
                        checkpoint_path: str) -> Optional[nn.Module]:
        """Load model checkpoint."""
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None