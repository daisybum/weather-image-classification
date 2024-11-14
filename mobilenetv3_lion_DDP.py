import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler

from utils.config import Config
from run.train import DistributedModelTrainer
from dataset_manager import DistributedDatasetManager
from model_manager import DistributedModelManager
from utils.setup import setup_distributed, cleanup_distributed, setup_logging
from run.model_factory import create_model
from run.optimizer_factory import create_optimizer, create_scheduler


def train_process(rank: int, world_size: int, config: Config):
    """
    Distributed training process to be run on each GPU.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        config (Config): Configuration object containing all training parameters
    """
    # 분산 학습 설정
    setup_distributed(rank, world_size)
    logger = setup_logging(rank)

    try:
        logger.info(f"Initializing process rank {rank}/{world_size - 1}")

        # 데이터셋 매니저 초기화
        dataset_manager = DistributedDatasetManager(
            train_path=config.dataset['train_path'],
            valid_path=config.dataset['valid_path'],
            batch_size=config.dataset['batch_size'],
            num_workers=config.dataset['num_workers']
        )

        # 데이터셋과 로더 준비
        logger.info("Preparing datasets and dataloaders...")
        train_dataset, val_dataset = dataset_manager.prepare_datasets(
            train_ratio=config.training['train_ratio']
        )
        train_loader, valid_loader, train_sampler, valid_sampler = \
            dataset_manager.create_dataloaders(
                train_dataset, val_dataset, rank, world_size
            )

        if rank == 0:
            logger.info(f"Total training samples: {len(train_dataset)}")
            logger.info(f"Total validation samples: {len(val_dataset)}")

        # 모델 매니저 초기화 및 모델 준비
        logger.info("Initializing model...")
        model_manager = DistributedModelManager(
            num_classes=config.model['num_classes'],
            rank=rank,
            sync_bn=config.model.get('sync_bn', True)
        )

        # 모델 생성
        model = create_model(
            model_name=config.model['name'],
            num_classes=config.model['num_classes'],
            pretrained=config.model.get('pretrained', True)
        )

        # 체크포인트에서 모델 로드 (있는 경우)
        if config.training.get('resume_from_checkpoint'):
            checkpoint_path = config.training['checkpoint_path']
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            model = model_manager.load_checkpoint(model, checkpoint_path)
            if model is None:
                logger.error("Failed to load checkpoint. Starting from scratch.")
                model = create_model(
                    model_name=config.model['name'],
                    num_classes=config.model['num_classes'],
                    pretrained=config.model.get('pretrained', True)
                )

        # 모델을 분산 학습용으로 준비
        model = model_manager.prepare_model(model)

        # Loss function 정의
        criterion = nn.CrossEntropyLoss().to(rank)

        # Optimizer 설정
        optimizer = create_optimizer(
            model=model,
            optimizer_name=config.training['optimizer'],
            learning_rate=config.training['learning_rate'],
            weight_decay=config.training['weight_decay']
        )

        # Learning rate scheduler 설정
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name=config.training['scheduler'],
            **config.training['scheduler_params']
        )

        # AMP scaler 초기화
        scaler = GradScaler()

        # Trainer 초기화
        trainer = DistributedModelTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device(f"cuda:{rank}"),
            save_path=config.paths['checkpoints'],
            model_name=f"{config.model['name']}_rank{rank}"
        )

        # Training 시작
        if rank == 0:
            logger.info("Starting training...")
            logger.info(f"Training configuration:")
            logger.info(f"- Epochs: {config.training['num_epochs']}")
            logger.info(f"- Batch size: {config.dataset['batch_size']}")
            logger.info(f"- Learning rate: {config.training['learning_rate']}")
            logger.info(f"- Optimizer: {config.training['optimizer']}")
            logger.info(f"- Scheduler: {config.training['scheduler']}")

        trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=config.training['num_epochs'],
            train_sampler=train_sampler,
            valid_sampler=valid_sampler
        )

        if rank == 0:
            logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in rank {rank}: {str(e)}", exc_info=True)
        raise  # Re-raise the exception for proper error handling
    finally:
        cleanup_distributed()
        logger.info(f"Process {rank} finished")


def main():
    # 환경 변수 로드
    load_dotenv()

    # 설정 로드
    config = Config()

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.training['gpus']))
    world_size = len(config.training['gpus'])

    try:
        # 분산 학습 시작
        mp.spawn(
            train_process,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"An error occurred in main process: {str(e)}")


if __name__ == '__main__':
    main()