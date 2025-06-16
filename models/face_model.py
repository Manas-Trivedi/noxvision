import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class FaceNet(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.5):
        super(FaceNet, self).__init__()

        # Use ResNet50 for better feature extraction
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # [B, 2048, 1, 1]

        # Freeze early layers to prevent overfitting
        for param in list(self.feature_extractor.parameters())[:50]:
            param.requires_grad = False

        # Flatten
        self.flatten = nn.Flatten()

        # More sophisticated embedding head with residual connection
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, embedding_dim)
        )

        # L2 normalization for cosine similarity
        self.l2_norm = True

        # Initialize weights
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
        # Extract features
        features = self.feature_extractor(x)
        features = self.flatten(features)

        # Get embeddings
        embeddings = self.embedding_head(features)

        # L2 normalize for cosine similarity
        if self.l2_norm:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def unfreeze_backbone(self):
        """Call this after some epochs to fine-tune the entire network"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True