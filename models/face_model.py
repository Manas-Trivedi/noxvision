import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math

class SelfAttention(nn.Module):
    """Self-attention mechanism for focusing on discriminative face regions."""
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Generate Q, K, V
        Q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        K = self.key(x).view(batch_size, -1, width * height)
        V = self.value(x).view(batch_size, -1, width * height)

        # Attention scores
        attention = torch.bmm(Q, K)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual connection with learnable gate
        out = self.gamma * out + x
        return out

class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class GeM(nn.Module):
    """Generalized Mean pooling."""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                           (x.size(-2), x.size(-1))).pow(1./self.p)

class EnhancedFaceNet(nn.Module):
    """Enhanced FaceNet with attention mechanisms and multi-scale features."""
    def __init__(self, embedding_dim=512, dropout=0.3, use_attention=True,
                 use_gem_pooling=True, use_multiscale=True):
        super(EnhancedFaceNet, self).__init__()

        self.use_attention = use_attention
        self.use_gem_pooling = use_gem_pooling
        self.use_multiscale = use_multiscale

        # Load pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Extract different layers for multi-scale features
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Freeze early layers initially
        for param in [self.conv1, self.bn1, self.layer1]:
            for p in param.parameters():
                p.requires_grad = False

        # Add attention modules
        if self.use_attention:
            self.attention1 = CBAM(256)
            self.attention2 = CBAM(512)
            self.attention3 = CBAM(1024)
            self.attention4 = CBAM(2048)
            self.self_attention = SelfAttention(2048, reduction=8)

        # Pooling
        if self.use_gem_pooling:
            self.global_pool = GeM(p=3)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Multi-scale feature fusion
        if self.use_multiscale:
            # Reduce channel dimensions for multi-scale features
            self.reduce1 = nn.Conv2d(256, 64, 1)
            self.reduce2 = nn.Conv2d(512, 128, 1)
            self.reduce3 = nn.Conv2d(1024, 256, 1)

            # Multi-scale pooling
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            self.pool2 = nn.AdaptiveAvgPool2d(1)
            self.pool3 = nn.AdaptiveAvgPool2d(1)

            # Total feature dimension: 2048 + 64 + 128 + 256 = 2496
            total_features = 2048 + 64 + 128 + 256
        else:
            total_features = 2048

        # Final embedding layers
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # L2 normalization for embeddings
        self.l2_norm = nn.functional.normalize

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of custom layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Multi-scale feature extraction
        features = []

        # Layer 1 - 256 channels
        x1 = self.layer1(x)
        if self.use_attention:
            x1 = self.attention1(x1)

        if self.use_multiscale:
            # Reduce and pool layer 1 features
            f1 = self.reduce1(x1)
            f1 = self.pool1(f1)
            f1 = f1.view(f1.size(0), -1)
            features.append(f1)

        # Layer 2 - 512 channels
        x2 = self.layer2(x1)
        if self.use_attention:
            x2 = self.attention2(x2)

        if self.use_multiscale:
            # Reduce and pool layer 2 features
            f2 = self.reduce2(x2)
            f2 = self.pool2(f2)
            f2 = f2.view(f2.size(0), -1)
            features.append(f2)

        # Layer 3 - 1024 channels
        x3 = self.layer3(x2)
        if self.use_attention:
            x3 = self.attention3(x3)

        if self.use_multiscale:
            # Reduce and pool layer 3 features
            f3 = self.reduce3(x3)
            f3 = self.pool3(f3)
            f3 = f3.view(f3.size(0), -1)
            features.append(f3)

        # Layer 4 - 2048 channels
        x4 = self.layer4(x3)
        if self.use_attention:
            x4 = self.attention4(x4)
            x4 = self.self_attention(x4)

        # Global pooling
        x4 = self.global_pool(x4)
        x4 = x4.view(x4.size(0), -1)
        features.append(x4)

        # Concatenate multi-scale features
        if self.use_multiscale and len(features) > 1:
            x = torch.cat(features, dim=1)
        else:
            x = x4

        # Final embedding
        x = self.embedding(x)

        # L2 normalize embeddings
        x = self.l2_norm(x, p=2, dim=1)

        return x

    def unfreeze_layers(self, unfreeze_layer2=False, unfreeze_layer1=False):
        """Gradually unfreeze layers for fine-tuning."""
        # Always allow layer3 and layer4 to be trainable
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True

        if unfreeze_layer2:
            for param in self.layer2.parameters():
                param.requires_grad = True

        if unfreeze_layer1:
            for param in self.layer1.parameters():
                param.requires_grad = True

    def get_embeddings(self, x):
        """Get normalized embeddings for inference."""
        return self.forward(x)


class SiameseNetwork(nn.Module):
    """Siamese Network for face verification using the enhanced backbone."""
    def __init__(self, embedding_dim=512, **kwargs):
        super(SiameseNetwork, self).__init__()
        self.backbone = EnhancedFaceNet(embedding_dim=embedding_dim, **kwargs)

    def forward(self, img1, img2):
        # Get embeddings for both images
        emb1 = self.backbone(img1)
        emb2 = self.backbone(img2)
        return emb1, emb2

    def forward_once(self, x):
        """Forward pass for single image."""
        return self.backbone(x)


class TripletNetwork(nn.Module):
    """Triplet Network for face verification."""
    def __init__(self, embedding_dim=512, **kwargs):
        super(TripletNetwork, self).__init__()
        self.backbone = EnhancedFaceNet(embedding_dim=embedding_dim, **kwargs)

    def forward(self, anchor, positive, negative):
        # Get embeddings for triplet
        emb_anchor = self.backbone(anchor)
        emb_positive = self.backbone(positive)
        emb_negative = self.backbone(negative)
        return emb_anchor, emb_positive, emb_negative

    def forward_once(self, x):
        """Forward pass for single image."""
        return self.backbone(x)


# Loss Functions
class ContrastiveLoss(nn.Module):
    """Contrastive Loss for Siamese networks."""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Calculate Euclidean distance
        distance = F.pairwise_distance(emb1, emb2, p=2)

        # Contrastive loss
        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet Loss with hard negative mining."""
    def __init__(self, margin=0.5, hard_mining=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Basic triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        if self.hard_mining:
            # Focus on hard examples (largest losses)
            hard_examples = loss > 0
            if hard_examples.sum() > 0:
                loss = loss[hard_examples]

        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    def __init__(self, triplet_weight=1.0, center_weight=0.1, margin=0.5):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight

    def forward(self, anchor, positive, negative):
        # Triplet loss
        triplet_loss = self.triplet_loss(anchor, positive, negative)

        # Center loss (encourage embeddings to be close to learned centers)
        center_loss = torch.mean(torch.norm(anchor, p=2, dim=1)) + \
                     torch.mean(torch.norm(positive, p=2, dim=1)) + \
                     torch.mean(torch.norm(negative, p=2, dim=1))

        total_loss = self.triplet_weight * triplet_loss + \
                    self.center_weight * center_loss

        return total_loss, triplet_loss, center_loss


# Model factory functions
def create_siamese_model(embedding_dim=512, use_attention=True, use_gem_pooling=True,
                        use_multiscale=True, dropout=0.3):
    """Create a Siamese network with enhanced backbone."""
    return SiameseNetwork(
        embedding_dim=embedding_dim,
        use_attention=use_attention,
        use_gem_pooling=use_gem_pooling,
        use_multiscale=use_multiscale,
        dropout=dropout
    )


def create_triplet_model(embedding_dim=512, use_attention=True, use_gem_pooling=True,
                        use_multiscale=True, dropout=0.3):
    """Create a Triplet network with enhanced backbone."""
    return TripletNetwork(
        embedding_dim=embedding_dim,
        use_attention=use_attention,
        use_gem_pooling=use_gem_pooling,
        use_multiscale=use_multiscale,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing Enhanced FaceNet...")
    model = EnhancedFaceNet(embedding_dim=512)
    model.to(device)

    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        embeddings = model(x)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding norm: {torch.norm(embeddings, p=2, dim=1)}")

    print("\nTesting Siamese Network...")
    siamese_model = create_siamese_model()
    siamese_model.to(device)

    img1 = torch.randn(2, 3, 224, 224).to(device)
    img2 = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        emb1, emb2 = siamese_model(img1, img2)
        print(f"Siamese embeddings shape: {emb1.shape}, {emb2.shape}")

    print("\nTesting Triplet Network...")
    triplet_model = create_triplet_model()
    triplet_model.to(device)

    anchor = torch.randn(2, 3, 224, 224).to(device)
    positive = torch.randn(2, 3, 224, 224).to(device)
    negative = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        emb_a, emb_p, emb_n = triplet_model(anchor, positive, negative)
        print(f"Triplet embeddings shape: {emb_a.shape}, {emb_p.shape}, {emb_n.shape}")

    print("\nModel created successfully! 🚀")