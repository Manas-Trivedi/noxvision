# models/gender_model.py

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class GenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)

        # Remove classification head and keep feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Output shape: (B, 512, 1, 1)
        self.flatten = nn.Flatten()

        # Gender classification head
        self.gender_head = nn.Linear(512, 2)

    def forward(self, x):
        features = self.flatten(self.backbone(x))  # Shape: (B, 512)
        return self.gender_head(features)
