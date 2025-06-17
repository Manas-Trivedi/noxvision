import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.face_model import FaceNet
from utils.face_dataset import EnhancedFaceRecognitionDataset  # Updated import
from utils.augmentations import get_val_transforms

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Calculate phi
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Calculate output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

def evaluate(model, arcface, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / len(val_loader)
    return acc, f1, avg_loss

def train():
    # Enhanced Config
    root = "data/facecom/Task_B"
    batch_size = 32  # Reduced for more stable training
    lr = 5e-5  # Lower learning rate
    epochs = 30  # More epochs
    embedding_dim = 512  # Larger embedding
    distortions_per_identity = 3
    synthetic_per_clean = 8  # More synthetic samples
    weight_decay = 1e-4

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Build identity mapping
    identity_folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
    identity_to_idx = {name: idx for idx, name in enumerate(identity_folders)}
    print(f"📊 Found {len(identity_folders)} identities")

    # Create enhanced datasets
    train_dataset = EnhancedFaceRecognitionDataset(
        root_dir=root,
        distortions_per_identity=distortions_per_identity,
        synthetic_per_clean=synthetic_per_clean,
        mode="train",
        image_size=224,
        identity_to_idx=identity_to_idx,
        balance_classes=True
    )

    val_dataset = EnhancedFaceRecognitionDataset(
        root_dir=root,
        distortions_per_identity=distortions_per_identity,
        synthetic_per_clean=0,  # No synthetic for validation
        mode="val",
        image_size=224,
        identity_to_idx=identity_to_idx,
        balance_classes=False
    )

    # Create weighted sampler for better class balance
    class_weights = train_dataset.get_class_weights()
    sample_weights = [float(class_weights[label]) for _, label, _ in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # Initialize model and loss
    model = FaceNet(embedding_dim=embedding_dim, dropout=0.5).to(device)
    arcface = ArcFaceLoss(
        embedding_dim=embedding_dim,
        num_classes=train_dataset.num_classes,
        s=30.0,  # Scale parameter
        m=0.5,   # Margin parameter
        easy_margin=False
    ).to(device)

    # Use Focal Loss for better handling of hard examples
    criterion = FocalLoss(alpha=1, gamma=2)

    # Optimizer with weight decay
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': lr},
        {'params': arcface.parameters(), 'lr': lr * 10}  # Higher LR for ArcFace
    ], weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Training metrics tracking
    train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []
    best_f1 = 0.0
    patience_counter = 0
    patience = 7

    print(f"🚀 Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    for epoch in range(1, epochs + 1):
        print(f"\n📦 Epoch {epoch}/{epochs}")

        # Refresh datasets to resample
        if epoch > 1:  # Skip first epoch
            train_dataset.refresh()
            val_dataset.refresh()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop with progress bar
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            embeddings = model(images)
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(arcface.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Statistics
            running_loss += loss.item()
            predicted = logits.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            current_acc = correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        # Calculate epoch metrics
        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        val_acc, val_f1, val_loss = evaluate(model, arcface, val_loader, device)

        # Store metrics
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        print(f"[Epoch {epoch:02d}] 📈 Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch:02d}] 🔍 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "face_model.pt")
            print(f"💾 New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"⏰ Early stopping triggered after {patience} epochs without improvement")
            break

        # Unfreeze backbone after a few epochs
        if epoch == 5:
            model.unfreeze_backbone()
            print("🔓 Unfroze backbone for fine-tuning")

    print(f"\n🎯 Training completed! Best F1 Score: {best_f1:.4f}")

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Val F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Set device-specific optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    train()