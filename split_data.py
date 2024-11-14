import os
import shutil
import random
from pathlib import Path


def split_dataset(source_root, train_root, eval_root, train_ratio=0.8):
    """
    source_root 디렉토리의 0-4 폴더에서 이미지를 불러와
    train_root와 eval_root의 각각의 클래스 폴더로 분배합니다.

    Args:
        source_root: 원본 데이터가 있는 루트 디렉토리
        train_root: 학습 데이터를 저장할 루트 디렉토리
        eval_root: 평가 데이터를 저장할 루트 디렉토리
        train_ratio: 학습 데이터의 비율 (기본값: 0.8)
    """
    # 필요한 디렉토리 생성
    for class_idx in range(5):  # 0부터 4까지
        Path(os.path.join(train_root, str(class_idx))).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(eval_root, str(class_idx))).mkdir(parents=True, exist_ok=True)

    # 각 클래스별로 이미지 분배
    for class_idx in range(5):
        source_dir = os.path.join(source_root, str(class_idx))

        # 해당 클래스의 모든 이미지 파일 리스트
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)  # 랜덤하게 섞기

        # train/val 분할 지점 계산
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        eval_files = image_files[split_idx:]

        # 학습 데이터 복사
        for file in train_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(train_root, str(class_idx), file)
            shutil.copy2(src, dst)

        # 평가 데이터 복사
        for file in eval_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(eval_root, str(class_idx), file)
            shutil.copy2(src, dst)

        print(f"Class {class_idx}:")
        print(f"  - Total images: {len(image_files)}")
        print(f"  - Train images: {len(train_files)}")
        print(f"  - Eval images: {len(eval_files)}")


# 사용 예시
if __name__ == "__main__":
    source_root = "C:\Projects\weather-condition-classification/data"  # 원본 데이터가 있는 루트 디렉토리
    train_root = "data/train"  # 학습 데이터를 저장할 디렉토리
    eval_root = "data/val"  # 평가 데이터를 저장할 디렉토리

    # 데이터 분할 실행
    split_dataset(source_root, train_root, eval_root, train_ratio=0.8)