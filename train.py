# train_balanced.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from models.gender_model import GenderNet
from utils.gender_dataset import BalancedGenderDataset
import numpy as np
import os

# 🧠 Improved Focal Loss with class weights
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def create_balanced_sampler(dataset):
    """Create a sampler that balances the classes"""
    labels = dataset.labels
    class_counts = Counter(labels)

    # Calculate weights for each class (inverse frequency)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}

    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def calculate_class_weights(dataset):
    """Calculate class weights for loss function"""
    labels = dataset.labels
    class_counts = Counter(labels)
    total_samples = len(labels)

    # Inverse frequency weighting
    weights = []
    for i in range(len(class_counts)):
        weight = total_samples / (len(class_counts) * class_counts[i])
        weights.append(weight)

    return torch.FloatTensor(weights)

def main():
    # 🔧 Configs
    train_path = "data/facecom/task_a/train"
    val_path = "data/facecom/task_a/val"
    batch_size = 32
    epochs = 20  # Increased epochs
    lr = 1e-4  # Lower learning rate for stability
    num_workers = 4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    save_path = "gender_model.pt"

    # 📦 Datasets
    train_dataset = BalancedGenderDataset(train_path, is_train=True)
    val_dataset = BalancedGenderDataset(val_path, is_train=False)

    print("Original Train distribution:", Counter(train_dataset.labels))
    print("Original Val distribution  :", Counter(val_dataset.labels))

    # 🎯 Create balanced sampler
    balanced_sampler = create_balanced_sampler(train_dataset)

    # 📊 Data loaders with balanced sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler,  # Use balanced sampler instead of shuffle
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 🧠 Model setup
    model = GenderNet().to(device)

    # 📏 Calculate class weights for loss function
    class_weights = calculate_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights}")

    # 🎯 Weighted Focal Loss
    loss_fn = WeightedFocalLoss(alpha=class_weights, gamma=2)

    # 🚀 Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 📈 Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_f1 = 0

    for epoch in range(epochs):
        # 🏃‍♂️ Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            gender_logits = model(images)
            loss = loss_fn(gender_logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Track training predictions
            preds = torch.argmax(gender_logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Print batch stats every 50 batches
            if batch_idx % 50 == 0:
                batch_acc = (preds == labels).float().mean()
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}")

        # 📊 Training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, zero_division=1)

        # ✅ Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                gender_logits = model(images)
                loss = loss_fn(gender_logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(gender_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 🧪 Validation metrics
        val_acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=1)
        recall = recall_score(all_labels, all_preds, zero_division=1)
        f1 = f1_score(all_labels, all_preds, zero_division=1)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"Val  : Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        print(f"Train Preds: {Counter(train_preds)}")
        print(f"Val Preds  : {Counter(all_preds)}")
        print(f"Val Labels : {Counter(all_labels)}")

        # Update learning rate
        scheduler.step(f1)

        # 💾 Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "gender_model.pt", _use_new_zipfile_serialization=False)
            print(f"✅ Saved new best model (F1: {best_f1:.4f}) to {save_path}")

        # Print detailed classification report every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\nDetailed Classification Report:")
            print(classification_report(all_labels, all_preds,
                                      target_names=['Male', 'Female']))
        print("-" * 80)

if __name__ == "__main__":
    main()