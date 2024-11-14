import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR, StepLR
from torchvision.models.efficientnet import efficientnet_b2
from torchvision.models import EfficientNet_B2_Weights
from lion_pytorch import Lion

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def augment_and_save_dataset(input_dir, output_dir, num_augmented_images_per_original=5):
    """
    Augment images in the dataset and save them to a new directory.

    Args:
    - input_dir (str): Directory with original images.
    - output_dir (str): Directory where augmented images will be saved.
    - num_augmented_images_per_original (int): Number of augmented images to generate per original image.
    """

    # Define transformations for data augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for root, _, filenames in os.walk(input_dir):
        for filename in tqdm(filenames):
            file_path = os.path.join(root, filename)
            image = Image.open(file_path)

            # Save multiple augmented versions of each image
            for i in range(num_augmented_images_per_original):
                transformed_image = transform(image)
                save_path = os.path.join(output_dir, root.split("\\")[-1], f"{filename.split('.')[0]}_aug{i}.jpg")
                transformed_image.save(save_path)


if __name__ == '__main__':
    # Load datasets
    origin_dir = 'D:/allbigdat/allbig_images/train'
    output_dir = 'D:/allbigdat/allbig_images/train_aug'

    augment_and_save_dataset(origin_dir, output_dir, num_augmented_images_per_original=5)
