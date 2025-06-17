import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from models.face_model import FaceNet

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

class FaceClassificationDataset(Dataset):
    """
    Dataset for Task B: Face Recognition Classification
    Loads distorted images and maps them to identity classes
    """
    def __init__(self, root_dir, transform=None, use_clean_images=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_clean_images = use_clean_images

        # Collect all samples
        self.samples = []
        self.identity_to_idx = {}
        self.idx_to_identity = {}

        identity_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        identity_folders.sort()  # Ensure consistent ordering

        for idx, identity_name in enumerate(identity_folders):
            self.identity_to_idx[identity_name] = idx
            self.idx_to_identity[idx] = identity_name

            identity_path = os.path.join(root_dir, identity_name)

            # Add clean reference image if requested
            if use_clean_images:
                clean_img_path = os.path.join(identity_path, f"{identity_name}.jpg")
                if os.path.exists(clean_img_path):
                    self.samples.append((clean_img_path, idx))

            # Add distorted images
            distortions_path = os.path.join(identity_path, "distortions")
            if os.path.exists(distortions_path):
                for img_file in os.listdir(distortions_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(distortions_path, img_file)
                        self.samples.append((img_path, idx))

        self.num_classes = len(self.identity_to_idx)
        print(f"📊 Loaded {len(self.samples)} samples from {self.num_classes} identities")

        # Print distribution
        class_counts = defaultdict(int)
        for _, class_idx in self.samples:
            class_counts[class_idx] += 1

        print(f"📈 Samples per class - Min: {min(class_counts.values())}, Max: {max(class_counts.values())}, Avg: {len(self.samples)/self.num_classes:.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, class_idx
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            # Return a random other sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

class FaceClassificationModel(nn.Module):
    """
    Face Recognition Model for 877-class classification
    """
    def __init__(self, num_classes, embedding_dim=512, dropout=0.5):
        super(FaceClassificationModel, self).__init__()

        # Use ResNet50 as backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)

        # Replace the final fully connected layer with a projection to embedding_dim
        self.backbone.fc = nn.Linear(2048, embedding_dim)

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )

        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def get_transforms():
    """Get data transforms for training and validation"""

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def calculate_metrics(model, dataloader, device):
    """Calculate accuracy and F1 score"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, f1

def train():
    # Configuration
    root_train = "data/facecom/Task_B/train"
    root_val = "data/facecom/Task_B/val"
    batch_size = 32
    lr = 1e-4  # Conservative learning rate
    epochs = 50
    embedding_dim = 512
    num_workers = 0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Data transforms
    train_transform, val_transform = get_transforms()

    # Load datasets
    print("📂 Loading training dataset...")
    train_dataset = FaceClassificationDataset(
        root_train,
        transform=train_transform,
        use_clean_images=True  # Include clean images for better training
    )

    print("📂 Loading validation dataset...")
    val_dataset = FaceClassificationDataset(
        root_val,
        transform=val_transform,
        use_clean_images=False  # Only distorted images for validation
    )

    # Ensure same number of classes
    if train_dataset.num_classes != val_dataset.num_classes:
        print(f"⚠️ Warning: Train classes ({train_dataset.num_classes}) != Val classes ({val_dataset.num_classes})")

    num_classes = train_dataset.num_classes
    print(f"🎯 Training for {num_classes} classes")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Model, loss, optimizer
    model = FaceClassificationModel(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps with generalization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0

    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("=" * 80)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Print progress
            if (batch_idx + 1) % 100 == 0:
                current_acc = 100.0 * train_correct / train_total
                print(f"    Batch {batch_idx+1:4d}/{len(train_loader):4d} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")

        # Calculate training metrics
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_acc, val_f1 = calculate_metrics(model, val_loader, device)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch results
        print(f"[Epoch {epoch+1:2d}/{epochs}] 📉 Loss: {avg_train_loss:.4f} | 🎯 Train Acc: {train_acc:.4f} | 🔍 Val Acc: {val_acc:.4f} | 📊 Val F1: {val_f1:.4f} | 📈 LR: {current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_f1

            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'embedding_dim': embedding_dim,
                'epoch': epoch,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'identity_to_idx': train_dataset.identity_to_idx,
                'idx_to_identity': train_dataset.idx_to_identity
            }, "best_face_classification_model.pt")

            # Also save just the state dict for easier loading
            torch.save(model.state_dict(), "face_classification_model.pt", _use_new_zipfile_serialization=False)

            print(f"💾 New best model saved! Val Acc: {best_val_acc:.4f}, Val F1: {best_val_f1:.4f}")

        print("-" * 80)

    print(f"\n🏆 Training completed!")
    print(f"📊 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"📊 Best Validation F1-Score: {best_val_f1:.4f}")

    # Final evaluation
    print("\n🔍 Final comprehensive evaluation...")
    final_acc, final_f1 = calculate_metrics(model, val_loader, device)
    print(f"📈 Final Metrics - Accuracy: {final_acc:.4f} | F1-Score: {final_f1:.4f}")

if __name__ == "__main__":
    # Set seeds for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train()