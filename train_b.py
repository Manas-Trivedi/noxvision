import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_auc_score
import seaborn as sns
from collections import defaultdict

# Import your custom modules
from utils.face_dataset import FaceVerificationDataset, FaceTripletDataset
from models.face_model import (
    create_siamese_model, create_triplet_model,
    ContrastiveLoss, TripletLoss, CombinedLoss
)

class FaceVerificationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'mps' if torch.backends.mps.is_available() else 'cpu'))
        print(f"🔧 Using device: {self.device}")

        # Set random seeds
        self.set_seeds(config.get('seed', 42))

        # Initialize datasets and models
        self.setup_datasets()
        self.setup_model()
        self.setup_training()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aucs = []
        self.best_auc = 0.0

    def set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setup_datasets(self):
        """Initialize datasets for verification training."""
        root_dir = self.config['root_dir']
        train_root = os.path.join(root_dir, "train")
        val_root = os.path.join(root_dir, "val")

        # ✅ Build identity mapping from BOTH train/ and val/
        identity_names = set()

        for split_root in [train_root, val_root]:
            if not os.path.exists(split_root):
                continue
            identity_names.update([
                f for f in os.listdir(split_root)
                if os.path.isdir(os.path.join(split_root, f))
            ])

        identity_folders = sorted(identity_names)
        self.identity_to_idx = {name: idx for idx, name in enumerate(identity_folders)}
        print(f"📊 Found {len(identity_folders)} total identities across train/val")

        # Create datasets based on training method
        if self.config['method'] == 'siamese':
            self.train_dataset = FaceVerificationDataset(
                root_dir=root_dir,
                mode="train",
                pairs_per_identity=self.config.get('pairs_per_identity', 50),
                hard_negative_ratio=self.config.get('hard_negative_ratio', 0.3),
                image_size=self.config.get('image_size', 224),
                identity_to_idx=self.identity_to_idx,
                synthetic_per_clean=self.config.get('synthetic_per_clean', 5)
            )

            self.val_dataset = FaceVerificationDataset(
                root_dir=root_dir,
                mode="val",
                pairs_per_identity=self.config.get('pairs_per_identity', 30),
                image_size=self.config.get('image_size', 224),
                identity_to_idx=self.identity_to_idx,
                synthetic_per_clean=0  # No synthetic for validation
            )

        elif self.config['method'] == 'triplet':
            self.train_dataset = FaceTripletDataset(
                root_dir=root_dir,
                mode="train",
                triplets_per_identity=self.config.get('triplets_per_identity', 30),
                image_size=self.config.get('image_size', 224),
                identity_to_idx=self.identity_to_idx,
                synthetic_per_clean=self.config.get('synthetic_per_clean', 5)
            )

            self.val_dataset = FaceTripletDataset(
                root_dir=root_dir,
                mode="val",
                triplets_per_identity=self.config.get('triplets_per_identity', 20),
                image_size=self.config.get('image_size', 224),
                identity_to_idx=self.identity_to_idx,
                synthetic_per_clean=0
            )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            drop_last=False
        )

        print(f"🚀 Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        print("Train dataset size:", len(self.train_dataset))

    def setup_model(self):
        """Initialize model and loss function."""
        model_config = {
            'embedding_dim': self.config.get('embedding_dim', 512),
            'use_attention': self.config.get('use_attention', True),
            'use_gem_pooling': self.config.get('use_gem_pooling', True),
            'use_multiscale': self.config.get('use_multiscale', True),
            'dropout': self.config.get('dropout', 0.3)
        }

        if self.config['method'] == 'siamese':
            self.model = create_siamese_model(**model_config)
            self.criterion = ContrastiveLoss(margin=self.config.get('margin', 1.0))

        elif self.config['method'] == 'triplet':
            self.model = create_triplet_model(**model_config)
            if self.config.get('use_combined_loss', False):
                self.criterion = CombinedLoss(
                    triplet_weight=1.0,
                    center_weight=0.1,
                    margin=self.config.get('margin', 0.5)
                )
            else:
                self.criterion = TripletLoss(
                    margin=self.config.get('margin', 0.5),
                    hard_mining=self.config.get('hard_mining', True)
                )

        self.model.to(self.device)
        print(f"🏗️ Model initialized: {self.config['method']} network")

    def setup_training(self):
        """Initialize optimizer and scheduler."""
        # Optimizer with different learning rates for backbone and head
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config.get('lr', 1e-4) * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.config.get('lr', 1e-4)}
        ], weight_decay=self.config.get('weight_decay', 1e-4))

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('lr', 1e-4),
            epochs=self.config.get('epochs', 25),
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        running_loss = 0.0

        # Refresh dataset pairs/triplets
        if epoch > 1:
            self.train_dataset.refresh()

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            if self.config['method'] == 'siamese':
                img1, img2, labels = batch_data
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                # Forward pass
                emb1, emb2 = self.model(img1, img2)
                loss = self.criterion(emb1, emb2, labels)

            elif self.config['method'] == 'triplet':
                anchor, positive, negative = batch_data
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Forward pass
                emb_a, emb_p, emb_n = self.model(anchor, positive, negative)

                if self.config.get('use_combined_loss', False):
                    loss, triplet_loss, center_loss = self.criterion(emb_a, emb_p, emb_n)
                else:
                    loss = self.criterion(emb_a, emb_p, emb_n)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate one epoch."""
        self.model.eval()
        running_loss = 0.0
        all_distances = []
        all_labels = []

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation", leave=False):
                if self.config['method'] == 'siamese':
                    img1, img2, labels = batch_data
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                    emb1, emb2 = self.model(img1, img2)
                    loss = self.criterion(emb1, emb2, labels)

                    # Calculate distances for evaluation
                    distances = F.pairwise_distance(emb1, emb2, p=2)
                    all_distances.extend(distances.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                elif self.config['method'] == 'triplet':
                    anchor, positive, negative = batch_data
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    emb_a, emb_p, emb_n = self.model(anchor, positive, negative)

                    if self.config.get('use_combined_loss', False):
                        loss, _, _ = self.criterion(emb_a, emb_p, emb_n)
                    else:
                        loss = self.criterion(emb_a, emb_p, emb_n)

                    # Calculate distances for evaluation
                    pos_distances = F.pairwise_distance(emb_a, emb_p, p=2)
                    neg_distances = F.pairwise_distance(emb_a, emb_n, p=2)

                    all_distances.extend(pos_distances.cpu().numpy())
                    all_labels.extend([1] * len(pos_distances))
                    all_distances.extend(neg_distances.cpu().numpy())
                    all_labels.extend([0] * len(neg_distances))

                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)

        # Calculate metrics
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)

        # For verification, closer distances should indicate same person (label=1)
        # So we need to flip the distances for AUC calculation
        similarity_scores = 1 / (1 + all_distances)  # Convert distance to similarity

        auc_score = roc_auc_score(all_labels, similarity_scores)

        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(all_labels, similarity_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5

        # Calculate accuracy with best threshold
        predictions = (similarity_scores >= best_threshold).astype(int)
        accuracy = accuracy_score(all_labels, predictions)

        return avg_loss, accuracy, auc_score, best_threshold

    def train(self):
        """Main training loop."""
        print(f"🚀 Starting {self.config['method']} training for {self.config.get('epochs', 25)} epochs")

        patience = self.config.get('patience', 7)
        patience_counter = 0

        for epoch in range(1, self.config.get('epochs', 25) + 1):
            print(f"\n📦 Epoch {epoch}/{self.config.get('epochs', 25)}")

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, val_auc, threshold = self.validate_epoch(epoch)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_aucs.append(val_auc)

            print(f"[Epoch {epoch:02d}] 📈 Train Loss: {train_loss:.4f}")
            print(f"[Epoch {epoch:02d}] 🔍 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            print(f"[Epoch {epoch:02d}] 🎯 Optimal Threshold: {threshold:.4f}")

            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                patience_counter = 0

                # Save model
                save_path = f"best_{self.config['method']}_model.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_auc': self.best_auc,
                    'threshold': threshold,
                    'config': self.config
                }, save_path)
                print(f"💾 New best model saved! AUC: {self.best_auc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"⏰ Early stopping triggered after {patience} epochs without improvement")
                break

            # Unfreeze backbone layers gradually
            if epoch == 5 and hasattr(self.model, 'backbone'):
                self.model.backbone.unfreeze_layers(unfreeze_layer2=True)
                print("🔓 Unfroze layer2 for fine-tuning")
            elif epoch == 10 and hasattr(self.model, 'backbone'):
                self.model.backbone.unfreeze_layers(unfreeze_layer2=True, unfreeze_layer1=True)
                print("🔓 Unfroze layer1 for fine-tuning")

        print(f"\n🎯 Training completed! Best AUC Score: {self.best_auc:.4f}")
        self.plot_training_curves()

    def plot_training_curves(self):
        """Plot training curves."""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(self.val_aucs, label='Val AUC', color='purple')
        plt.title('Validation AUC Score')
        plt.xlabel('Epoch')
        plt.ylabel('AUC Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.config["method"]}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration for training
    config = {
        # Data
        'root_dir': "data/facecom/Task_B",
        'image_size': 224,
        'pairs_per_identity': 50,  # For Siamese
        'triplets_per_identity': 30,  # For Triplet
        'synthetic_per_clean': 5,
        'hard_negative_ratio': 0.3,

        # Model
        'method': 'triplet',  # 'siamese' or 'triplet'
        'embedding_dim': 512,
        'use_attention': True,
        'use_gem_pooling': True,
        'use_multiscale': True,
        'dropout': 0.3,

        # Training
        'batch_size': 24,  # Adjusted for memory
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'epochs': 25,
        'patience': 7,

        # Loss
        'margin': 0.5,  # For both contrastive and triplet loss
        'hard_mining': True,  # For triplet loss
        'use_combined_loss': True,  # For triplet loss

        # System
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'num_workers': 4,
        'seed': 42
    }

    print("=" * 60)
    print(f"🎯 Face Verification Training - {config['method'].upper()} Method")
    print("=" * 60)

    # Create trainer and start training
    trainer = FaceVerificationTrainer(config)
    trainer.train()

    # Also train with Siamese method for comparison
    if config['method'] == 'triplet':
        print("\n" + "=" * 60)
        print("🎯 Training Siamese Network for Comparison")
        print("=" * 60)

        siamese_config = config.copy()
        siamese_config['method'] = 'siamese'
        siamese_config['margin'] = 1.0  # Different margin for contrastive loss

        siamese_trainer = FaceVerificationTrainer(siamese_config)
        siamese_trainer.train()

if __name__ == "__main__":
    main()