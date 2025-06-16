import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import random
from collections import defaultdict
from models.face_model import FaceNet
from utils.face_dataset import FaceRecognitionDataset
from utils.augmentations import get_val_transforms, get_aggressive_train_transforms

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Creates triplets (anchor, positive, negative) for triplet loss training
    """
    def __init__(self, base_dataset, samples_per_identity=5):
        self.base_dataset = base_dataset
        self.samples_per_identity = samples_per_identity

        # Group samples by identity
        self.identity_to_samples = defaultdict(list)
        for idx, (path, identity) in enumerate(base_dataset.samples):
            self.identity_to_samples[identity].append(idx)

        # Filter identities with at least 2 samples (need positive pairs)
        self.valid_identities = [
            identity for identity, samples in self.identity_to_samples.items()
            if len(samples) >= 2
        ]

        print(f"🔄 Created triplet dataset with {len(self.valid_identities)} identities")
        print(f"📊 Total samples available: {len(base_dataset.samples)}")

    def __len__(self):
        return len(self.valid_identities) * self.samples_per_identity

    def __getitem__(self, idx):
        # Select anchor identity
        identity_idx = idx // self.samples_per_identity
        anchor_identity = self.valid_identities[identity_idx]

        # Get anchor and positive from same identity
        identity_samples = self.identity_to_samples[anchor_identity]
        anchor_idx, positive_idx = random.sample(identity_samples, 2)

        # Get negative from different identity
        negative_identity = random.choice([id for id in self.valid_identities if id != anchor_identity])
        negative_idx = random.choice(self.identity_to_samples[negative_identity])

        # Load images
        anchor_img, _ = self.base_dataset[anchor_idx]
        positive_img, _ = self.base_dataset[positive_idx]
        negative_img, _ = self.base_dataset[negative_idx]

        return anchor_img, positive_img, negative_img, anchor_identity

class OnlineTripletLoss(nn.Module):
    """
    Fixed Online triplet loss - mines hard triplets within each batch
    """
    def __init__(self, margin=0.3):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Calculate pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(1)
        pos_mask = labels == labels.t()
        neg_mask = labels != labels.t()

        # Remove diagonal (self-comparisons)
        pos_mask.fill_diagonal_(False)

        # Find hardest positive (furthest positive pair)
        # Set impossible distances for non-positive pairs to -inf, so max() ignores them
        pos_dist = pairwise_dist.clone()
        pos_dist[~pos_mask] = -float('inf')

        # Get the maximum positive distance for each anchor (hardest positive)
        hardest_pos_dist, _ = pos_dist.max(dim=1)

        # Handle case where no positive pairs exist
        valid_pos_mask = hardest_pos_dist != -float('inf')
        if not valid_pos_mask.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Find hardest negative (closest negative pair)
        # Set impossible distances for non-negative pairs to +inf, so min() ignores them
        neg_dist = pairwise_dist.clone()
        neg_dist[~neg_mask] = float('inf')

        # Get the minimum negative distance for each anchor (hardest negative)
        hardest_neg_dist, _ = neg_dist.min(dim=1)

        # Handle case where no negative pairs exist
        valid_neg_mask = hardest_neg_dist != float('inf')
        valid_mask = valid_pos_mask & valid_neg_mask

        if not valid_mask.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Calculate triplet loss only for valid triplets
        hardest_pos_dist = hardest_pos_dist[valid_mask]
        hardest_neg_dist = hardest_neg_dist[valid_mask]

        # Calculate triplet loss
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        return loss.mean()

def create_verification_pairs(dataset, num_pairs=1000):
    """
    Create positive and negative pairs for verification evaluation
    """
    identity_to_samples = defaultdict(list)
    for idx, (path, identity) in enumerate(dataset.samples):
        identity_to_samples[identity].append(idx)

    positive_pairs = []
    negative_pairs = []

    identities = list(identity_to_samples.keys())

    # Create positive pairs (same identity)
    for identity, samples in identity_to_samples.items():
        if len(samples) >= 2:
            for i in range(min(num_pairs // len(identities), len(samples))):
                if len(samples) >= 2:
                    idx1, idx2 = random.sample(samples, 2)
                    positive_pairs.append((idx1, idx2, 1))  # Label 1 for same person

    # Create negative pairs (different identities)
    for _ in range(len(positive_pairs)):
        id1, id2 = random.sample(identities, 2)
        idx1 = random.choice(identity_to_samples[id1])
        idx2 = random.choice(identity_to_samples[id2])
        negative_pairs.append((idx1, idx2, 0))  # Label 0 for different person

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    print(f"🔄 Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
    return all_pairs

def evaluate_verification(model, dataset, device, num_pairs=1000):
    """
    Evaluate face verification performance
    """
    model.eval()
    pairs = create_verification_pairs(dataset, num_pairs)

    if len(pairs) == 0:
        return 0.0, 0.0, 0.0

    similarities = []
    labels = []

    with torch.no_grad():
        for idx1, idx2, label in pairs:
            # Get images
            img1, _ = dataset[idx1]
            img2, _ = dataset[idx2]

            # Get embeddings
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)

            emb1 = model(img1)
            emb2 = model(img2)

            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
            similarities.append(similarity)
            labels.append(label)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find best threshold using validation data
    thresholds = np.linspace(0.0, 1.0, 100)
    best_acc = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        acc = accuracy_score(labels, predictions)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Calculate final metrics with best threshold
    final_predictions = (similarities > best_threshold).astype(int)
    accuracy = accuracy_score(labels, final_predictions)
    f1 = f1_score(labels, final_predictions, average='macro')

    try:
        auc = roc_auc_score(labels, similarities)
    except:
        auc = 0.0

    return accuracy, f1, auc

def train():
    # Config
    root_train = "data/facecom/Task_B/train"
    root_val = "data/facecom/Task_B/val"
    batch_size = 32
    lr = 5e-5  # Lower learning rate for stable training
    epochs = 30  # More epochs for triplet learning
    embedding_dim = 256  # Larger embedding for better representation
    num_workers = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Load datasets
    train_base_ds = FaceRecognitionDataset(root_train, transform=get_aggressive_train_transforms())
    val_ds = FaceRecognitionDataset(root_val, transform=get_val_transforms())

    # Create triplet dataset
    train_triplet_ds = TripletDataset(train_base_ds, samples_per_identity=10)

    print(f"📊 Training triplets: {len(train_triplet_ds)}")
    print(f"📊 Validation samples: {len(val_ds.samples)}")

    # Create data loader for triplets
    def triplet_collate_fn(batch):
        anchors, positives, negatives, identities = zip(*batch)
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        # Create labels for online triplet mining (we'll use identity indices)
        unique_identities = list(set(identities))
        id_to_idx = {identity: idx for idx, identity in enumerate(unique_identities)}
        labels = torch.tensor([id_to_idx[identity] for identity in identities])

        # Combine all images and create corresponding labels
        all_images = torch.cat([anchors, positives, negatives], dim=0)
        all_labels = torch.cat([labels, labels, torch.arange(len(labels)) + len(unique_identities)], dim=0)

        return all_images, all_labels

    train_loader = DataLoader(
        train_triplet_ds,
        batch_size=batch_size//3,  # Smaller batch size since we triple the data
        shuffle=True,
        num_workers=num_workers,
        collate_fn=triplet_collate_fn
    )

    # Model and loss
    model = FaceNet(embedding_dim=embedding_dim).to(device)
    triplet_loss = OnlineTripletLoss(margin=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    # Add learning rate warmup
    warmup_epochs = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            embeddings = model(images)
            loss = triplet_loss(embeddings, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"    Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        # Evaluate every 3 epochs to save time
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            val_acc, val_f1, val_auc = evaluate_verification(model, val_ds, device, num_pairs=500)

            print(f"[Epoch {epoch+1:2d}/{epochs}] 📉 Loss: {avg_loss:.4f} | 🎯 Val Acc: {val_acc:.4f} | 📊 Val F1: {val_f1:.4f} | 🔄 AUC: {val_auc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "face_model.pt", _use_new_zipfile_serialization=False)
                print(f"💾 New best model saved! Acc: {best_val_acc:.4f}")
        else:
            print(f"[Epoch {epoch+1:2d}/{epochs}] 📉 Loss: {avg_loss:.4f}")

    print(f"🏆 Training completed! Best validation accuracy: {best_val_acc:.4f}")

    # Final comprehensive evaluation
    print("\n🔍 Final Evaluation:")
    final_acc, final_f1, final_auc = evaluate_verification(model, val_ds, device, num_pairs=2000)
    print(f"📊 Final Metrics - Acc: {final_acc:.4f} | F1: {final_f1:.4f} | AUC: {final_auc:.4f}")

if __name__ == "__main__":
    train()