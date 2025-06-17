import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class FaceNet(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.3):
        super(FaceNet, self).__init__()

        # Load pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B, 2048, 1, 1]

        # Freeze early layers to preserve pretrained stability (can unfreeze later)
        for param in list(self.backbone.parameters())[:50]:
            param.requires_grad = False

        # Flatten
        self.flatten = nn.Flatten()

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Apply weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)           # [B, 2048, 1, 1]
        x = self.flatten(x)            # [B, 2048]
        x = self.embedding_head(x)     # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1) # L2-normalize for ArcFace
        return x

    def unfreeze_backbone(self):
        """Unfreeze the ResNet backbone (used after a few epochs)."""
        for param in self.backbone.parameters():
            param.requires_grad = True