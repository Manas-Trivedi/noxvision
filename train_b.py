import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import random
from PIL import Image
from collections import defaultdict
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.face_model import FaceNet
from utils.augmentations import get_balanced_train_transforms, get_val_transforms

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

def setup_logging(log_dir="logs"):
    """Setup comprehensive logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)

    # Create timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"face_classification_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting face classification training - Log file: {log_file}")
    return logger

class TrainingMetrics:
    """Class to track and save training metrics"""
    def __init__(self, save_dir="results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_f1s = []
        self.learning_rates = []

    def update(self, train_loss, train_acc, val_loss, val_acc, val_f1, lr):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.val_f1s.append(val_f1)
        self.learning_rates.append(lr)

    def save_metrics(self, filename="training_metrics.json"):
        metrics = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accs,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accs,
            'val_f1_scores': self.val_f1s,
            'learning_rates': self.learning_rates
        }

        with open(self.save_dir / filename, 'w') as f:
            json.dump(metrics, f, indent=2)

    def plot_metrics(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.train_accs, label='Train Acc', color='blue')
        axes[0, 1].plot(self.val_accs, label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score
        axes[1, 0].plot(self.val_f1s, label='Val F1', color='green')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning Rate
        axes[1, 1].plot(self.learning_rates, label='Learning Rate', color='orange')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

class FaceClassificationDataset(Dataset):
    def __init__(self, root_dirs, transform=None, use_clean_images=False, logger=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.transform = transform
        self.use_clean_images = use_clean_images
        self.logger = logger or logging.getLogger(__name__)

        self.samples = []
        self.identity_to_idx = {}
        self.idx_to_identity = {}
        idx = 0

        for root_dir in root_dirs:
            self.logger.info(f"📂 Processing directory: {root_dir}")
            if not os.path.exists(root_dir):
                self.logger.warning(f"⚠️ Directory not found: {root_dir}")
                continue

            identity_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.logger.info(f"Found {len(identity_folders)} identity folders")

            for identity_name in identity_folders:
                if identity_name in self.identity_to_idx:
                    continue

                self.identity_to_idx[identity_name] = idx
                self.idx_to_identity[idx] = identity_name
                identity_path = os.path.join(root_dir, identity_name)

                samples_for_identity = 0

                if use_clean_images:
                    clean_img = os.path.join(identity_path, f"{identity_name}.jpg")
                    if os.path.exists(clean_img):
                        self.samples.append((clean_img, idx))
                        samples_for_identity += 1

                distortions = os.path.join(identity_path, "distortion")
                if os.path.exists(distortions):
                    distortion_files = [f for f in os.listdir(distortions) if f.endswith((".jpg", ".jpeg", ".png"))]
                    for f in distortion_files:
                        self.samples.append((os.path.join(distortions, f), idx))
                        samples_for_identity += 1

                if samples_for_identity > 0:
                    self.logger.debug(f"Identity '{identity_name}': {samples_for_identity} samples")
                    idx += 1

        self.num_classes = len(self.identity_to_idx)
        self.logger.info(f"📊 Dataset loaded: {len(self.samples)} samples from {self.num_classes} identities")

        # Log class distribution
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[label] += 1

        self.logger.info(f"📈 Class distribution - Min: {min(class_counts.values())}, Max: {max(class_counts.values())}, Mean: {np.mean(list(class_counts.values())):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            if self.transform:
                image = self.transform(image=image)['image']
            return image, label
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading {img_path}: {e}")
            # Return a random sample instead
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

def stratified_split(samples, val_per_class=1, logger=None):
    """Create stratified train/val split"""
    if logger is None:
        logger = logging.getLogger(__name__)

    class_to_indices = defaultdict(list)
    for i, (_, cls) in enumerate(samples):
        class_to_indices[cls].append(i)

    train_idx, val_idx = [], []
    classes_with_insufficient_samples = 0

    for cls, indices in class_to_indices.items():
        random.shuffle(indices)
        if len(indices) < val_per_class:
            classes_with_insufficient_samples += 1
            # If not enough samples, put all in training
            train_idx += indices
        else:
            val_idx += indices[:val_per_class]
            train_idx += indices[val_per_class:]

    if classes_with_insufficient_samples > 0:
        logger.warning(f"⚠️ {classes_with_insufficient_samples} classes have fewer than {val_per_class} samples")

    logger.info(f"📊 Split created: {len(train_idx)} train, {len(val_idx)} validation samples")
    return train_idx, val_idx

def save_model_checkpoint(model, logger=None):
    """Save comprehensive model checkpoint"""
    if logger is None:
        logger = logging.getLogger(__name__)

    torch.save(model.state_dict(), "face_model.pt", _use_new_zipfile_serialization=False)

    logger.info(f"💾 Checkpoint saved")

def evaluate_model(model, val_loader, criterion, device, dataset, logger=None):
    """Comprehensive model evaluation"""
    if logger is None:
        logger = logging.getLogger(__name__)

    model.eval()
    val_loss, val_preds, val_labels = 0.0, [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(labels.cpu().tolist())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")

    # Detailed classification report
    if len(set(val_labels)) > 2:  # Multi-class
        class_report = classification_report(
            val_labels, val_preds,
            target_names=[dataset.idx_to_identity[i] for i in sorted(set(val_labels))],
            output_dict=True
        )
        logger.info(f"📊 Detailed Classification Metrics:")
        logger.info(f"   Macro F1: {class_report.get('macro avg', {}).get('f1-score', 0.0):.4f}")
        logger.info(f"   Weighted F1: {class_report.get('weighted avg', {}).get('f1-score', 0.0):.4f}")

    return val_loss, val_acc, val_f1, val_preds, val_labels

def train():
    # Setup logging
    logger = setup_logging()

    # Configuration
    config = {
        'root_train': "data/facecom/Task_B/train",
        'root_val': "data/facecom/Task_B/val",
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 50,
        'embedding_dim': 512,
        'patience_to_unfreeze': 5,
        'early_stopping_patience': 10,
        'val_per_class': 1
    }

    logger.info(f"🔧 Training Configuration: {json.dumps(config, indent=2)}")

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")

    # Data transforms
    train_transform = get_balanced_train_transforms()
    val_transform = get_val_transforms()
    logger.info("🔄 Data transformations configured")

    # Dataset creation
    logger.info("📂 Loading datasets...")
    dataset = FaceClassificationDataset(
        [config['root_train'], config['root_val']],
        transform=train_transform,
        use_clean_images=True,
        logger=logger
    )

    # Train/val split
    train_idx, val_idx = stratified_split(dataset.samples, config['val_per_class'], logger)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(
        FaceClassificationDataset(
            [config['root_train'], config['root_val']],
            transform=val_transform,
            use_clean_images=True,
            logger=logger
        ),
        val_idx
    )

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"📊 Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Model setup
    model = FaceNet(embedding_dim=config['embedding_dim']).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    logger.info(f"🧠 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"🎯 Training {dataset.num_classes} classes")

    # Training tracking
    metrics = TrainingMetrics()
    best_val_loss = float('inf')
    best_val_acc = 0.0
    stagnation_counter = 0
    early_stopping_counter = 0
    backbone_unfrozen = False

    logger.info("🚀 Starting training...")

    for epoch in range(1, config['epochs'] + 1):
        logger.info(f"🔄 Epoch {epoch}/{config['epochs']}")

        # Training phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            total_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct/total:.4f}'
                })

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate_model(
            model, val_loader, criterion, device, dataset, logger
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update metrics
        metrics.update(train_loss, train_acc, val_loss, val_acc, val_f1, current_lr)

        # Logging
        logger.info(f"[Epoch {epoch:02d}] ✅ Train Loss: {train_loss:.4f} | 🎯 Train Acc: {train_acc:.4f}")
        logger.info(f"[Epoch {epoch:02d}] 🔍 Val Loss: {val_loss:.4f} | 🎯 Val Acc: {val_acc:.4f} | 📊 Val F1: {val_f1:.4f}")
        logger.info(f"[Epoch {epoch:02d}] 📈 Learning Rate: {current_lr:.2e}")

        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            stagnation_counter = 0
            early_stopping_counter = 0

            # Save best model
            save_model_checkpoint(
                model, logger
            )
            logger.info("💾 New best model saved!")
        else:
            stagnation_counter += 1
            early_stopping_counter += 1

        # Progressive unfreezing
        if stagnation_counter >= config['patience_to_unfreeze'] and not backbone_unfrozen:
            logger.info("🔓 Unfreezing backbone due to loss plateau...")
            model.unfreeze_backbone()
            backbone_unfrozen = True
            stagnation_counter = 0  # Reset counter after unfreezing

        # Early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            logger.info(f"⏹️ Early stopping triggered after {epoch} epochs")
            break

    # Final results
    logger.info("🎉 Training completed!")
    logger.info(f"📊 Best validation loss: {best_val_loss:.4f}")
    logger.info(f"🎯 Best validation accuracy: {best_val_acc:.4f}")

    # Save metrics and plots
    metrics.save_metrics()
    metrics.plot_metrics()
    logger.info("📈 Training curves saved to results/training_curves.png")

    # Save final model
    save_model_checkpoint(
        model, logger
    )

    logger.info("✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train()