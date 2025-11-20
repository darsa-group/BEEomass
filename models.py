import torchvision.models as models
import torch.nn as nn

# -------------------- MODEL --------------------

def build_resnet101(pretrained: bool = True):
    model = models.resnet101(pretrained=pretrained)
    # Replace final fc with one output (regression)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def build_resnet18(pretrained: bool = True):
    model = models.resnet18(pretrained=pretrained)
    # Replace final fc with one output (regression)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model