import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """모델 생성 팩토리 함수"""
    if model_name == "mobilenetv3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")