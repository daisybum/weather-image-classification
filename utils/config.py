import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # Get base path from environment variable or use default
        base_path = os.getenv('DATASET_PATH', self._config['dataset']['base_path'])
        self._config['dataset']['base_path'] = base_path

        # Setup paths
        self.setup_paths()

    def setup_paths(self):
        """Convert configured paths to absolute paths and verify existence"""
        base_path = Path(self._config['dataset']['base_path'])

        # Setup training/validation data paths
        self._config['dataset']['train_path'] = str(base_path / self._config['dataset']['train_dir'])
        self._config['dataset']['valid_path'] = str(base_path / self._config['dataset']['valid_dir'])

        # Setup checkpoint and log paths
        self._config['paths']['checkpoints'] = str(Path(self._config['paths']['checkpoints']).absolute())
        self._config['paths']['logs'] = str(Path(self._config['paths']['logs']).absolute())

        # Create necessary directories
        os.makedirs(self._config['paths']['checkpoints'], exist_ok=True)
        os.makedirs(self._config['paths']['logs'], exist_ok=True)

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def training(self):
        return self._config['training']

    @property
    def model(self):
        return self._config['model']

    @property
    def paths(self):
        return self._config['paths']