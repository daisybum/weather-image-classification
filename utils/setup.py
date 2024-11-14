import os
import torch
import torch.distributed as dist
import logging
from pathlib import Path


def setup_distributed(rank: int, world_size: int, master_port: str = '12355') -> None:
    """Initialize distributed training setup."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    dist.destroy_process_group()


def setup_logging(rank: int) -> logging.Logger:
    """Set up logging configuration for distributed training."""
    logger = logging.getLogger(f'Trainer-{rank}')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 파일 핸들러 설정
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"training_gpu_{rank}.log")
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

        # 콘솔 핸들러는 마스터 프로세스에만 추가
        if rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(console_handler)

    return logger