# models/model.py

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class NoxVisionNet(nn.Module):
    def __init__(self, num_classes_identity=100):
        super(NoxVisionNet, self).__init__()
        # Load ResNet18 backbone with new weights API
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        # Remove the last fully connected layer from ResNet18
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # output: [batch, 512, 1, 1]

        # Flatten output of backbone
        self.flatten = nn.Flatten()

        # Gender classification head (binary classification)
        self.gender_head = nn.Linear(512, 2)

        # Identity classification head (multi-class)
        self.identity_head = nn.Linear(512, num_classes_identity)

    def forward(self, x):
        features = self.backbone(x)
        features = self.flatten(features)

        gender_logits = self.gender_head(features)
        identity_logits = self.identity_head(features)

        return gender_logits, identity_logits
