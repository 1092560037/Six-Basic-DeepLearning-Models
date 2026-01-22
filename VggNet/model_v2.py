import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


def vgg(model_name="vgg16", num_classes=5, weights=VGG16_Weights.DEFAULT):
    assert model_name == "vgg16", f"Only vgg16 supported, got {model_name}"
    model = models.vgg16(weights=weights)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model