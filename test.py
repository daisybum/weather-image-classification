import os
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models.efficientnet import efficientnet_b2, efficientnet_v2_s
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def test_transform():
    """Transforms for the test dataset."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def old_transform():
    """Transforms for the validation dataset."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate_batch(model, batch, device):
    with torch.no_grad():
        model.eval()
        outputs = model(batch.to(device))
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


def load_images(image_paths, transform):
    images = [transform(Image.open(path).convert('RGB')) for path in image_paths]
    return torch.stack(images)


if __name__ == '__main__':
    # Load the trained model
    model_path = 'model/model_epoch_28.pth'  # Update this path
    model = mobilenet_v3_small(weights=None)
    num_classes = 5
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Class names (update as per your dataset)
    class_names = ['0', '1', '2', '3', '4']

    # Transform
    transform = test_transform()

    # Directory containing images
    image_directory = './data/val/*'  # Update this path
    image_paths = glob(os.path.join(image_directory, '*.jpg'))

    # Batch size
    batch_size = 64  # You can adjust this based on your memory constraints

    # Evaluate images in batches
    results = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch = load_images(batch_paths, transform)
        predicted_classes = evaluate_batch(model, batch, device)
        for path, pred in zip(batch_paths, predicted_classes):
            results.append({'Image Path': path, 'Predicted Class': class_names[pred]})

    df = pd.DataFrame(results)
    df['Class Name'] = df['Image Path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df['result'] = (df['Class Name'] == df['Predicted Class'])

    precision = df['result'].sum() / len(df)
    print(precision)
