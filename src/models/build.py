from torchvision.models import resnet18
from torch import nn


def build_model(device):
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(512, 6)

    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)
    return model

