import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from utils.augmentations import get_balanced_train_transforms, get_aggressive_train_transforms, get_val_transforms

class EnhancedFaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train",
                 distortions_per_identity=3, synthetic_per_clean=5,
                 image_size=224, identity_to_idx=None,
                 balance_classes=True):
        """
        Enhanced dataset that creates synthetic distortions from clean images

        Args:
            root_dir (str): Path to Task_B directory containing identity folders.
            transform: Albumentations transform to apply.
            mode (str): 'train' or 'val'
            distortions_per_identity (int): How many existing distorted images to sample per identity per epoch.
            synthetic_per_clean (int): How many synthetic distortions to create per clean image
            image_size (int): Target image size
            identity_to_idx (dict): Mapping from identity names to indices
            balance_classes (bool): Whether to balance classes by oversampling minority classes
        """
        assert mode in {"train", "val"}, "Mode must be 'train' or 'val'"
        self.root_dir = root_dir
        self.mode = mode
        self.distortions_per_identity = distortions_per_identity
        self.synthetic_per_clean = synthetic_per_clean
        self.image_size = image_size
        self.balance_classes = balance_classes

        # Set up transforms
        if transform is None:
            if mode == "train":
                self.transform = get_balanced_train_transforms(image_size)
                self.synthetic_transform = get_aggressive_train_transforms(image_size)
            else:
                self.transform = get_val_transforms(image_size)
                self.synthetic_transform = get_val_transforms(image_size)
        else:
            self.transform = transform
            self.synthetic_transform = transform

        self.samples = []
        self.identity_to_idx = identity_to_idx or {}
        self.clean_images_per_identity = {}  # Store clean images for synthetic generation
        self._build_dataset()

    def _build_dataset(self):
        identity_folders = sorted(os.listdir(self.root_dir))
        if not self.identity_to_idx:
            self.identity_to_idx = {name: idx for idx, name in enumerate(identity_folders)
                                  if os.path.isdir(os.path.join(self.root_dir, name))}

        self.samples.clear()
        self.clean_images_per_identity.clear()
        identity_sample_counts = {}

        for identity_name, idx in self.identity_to_idx.items():
            identity_path = os.path.join(self.root_dir, identity_name)
            clean_images = []
            sample_count = 0

            # 1. Collect all clean images in the identity folder (not in 'distortion')
            for file in os.listdir(identity_path):
                file_path = os.path.join(identity_path, file)
                if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
                    clean_images.append(file_path)
                    self.samples.append((file_path, idx, "clean"))
                    sample_count += 1

            # Store clean images for synthetic generation
            self.clean_images_per_identity[idx] = clean_images

            # 2. Add synthetic distortions from clean images (only in train mode)
            if self.mode == "train" and clean_images:
                for clean_path in clean_images:
                    for i in range(self.synthetic_per_clean):
                        self.samples.append((clean_path, idx, f"synthetic_{i}"))
                        sample_count += 1

            # 3. Add existing distortions
            distortion_dir = os.path.join(identity_path, "distortion")
            if os.path.isdir(distortion_dir):
                distortions = [f for f in os.listdir(distortion_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                if len(distortions) >= 2:
                    random.shuffle(distortions)
                    midpoint = len(distortions) // 2

                    if self.mode == "train":
                        selected = distortions[:midpoint][:self.distortions_per_identity]
                    else:
                        selected = distortions[midpoint:][:self.distortions_per_identity]

                    for file in selected:
                        path = os.path.join(distortion_dir, file)
                        self.samples.append((path, idx, "distortion"))
                        sample_count += 1

            identity_sample_counts[idx] = sample_count

        # Balance classes if requested (only in train mode)
        if self.mode == "train" and self.balance_classes:
            self._balance_classes(identity_sample_counts)

        self.num_classes = len(self.identity_to_idx)
        print(f"[{self.mode.upper()}] ✅ Loaded {len(self.samples)} samples from {len(self.identity_to_idx)} identities.")

        # Print class distribution
        if len(self.samples) > 0:
            class_counts = {}
            for _, label, _ in self.samples:
                class_counts[label] = class_counts.get(label, 0) + 1
            print(f"[{self.mode.upper()}] 📊 Class distribution: min={min(class_counts.values())}, "
                  f"max={max(class_counts.values())}, avg={np.mean(list(class_counts.values())):.1f}")

    def _balance_classes(self, identity_sample_counts):
        """Balance classes by oversampling underrepresented identities"""
        if not identity_sample_counts:
            return

        max_samples = max(identity_sample_counts.values())
        target_samples = int(max_samples * 0.2)  # Try 20% of max class size, or even less

        additional_samples = []
        for identity_idx, current_count in identity_sample_counts.items():
            if current_count < target_samples:
                needed = target_samples - current_count
                # Find samples for this identity
                identity_samples = [(path, label, sample_type) for path, label, sample_type in self.samples
                                  if label == identity_idx]

                # Oversample by duplicating existing samples
                for _ in range(needed):
                    if identity_samples:
                        sampled = random.choice(identity_samples)
                        additional_samples.append(sampled)

        self.samples.extend(additional_samples)
        print(f"[{self.mode.upper()}] ⚖️ Added {len(additional_samples)} samples for class balancing")

    def _load_and_preprocess_image(self, image_path):
        """Load image and apply basic preprocessing"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

            # Basic preprocessing: resize if too large
            h, w = image.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))

            return image
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, sample_type = self.samples[idx]

        image = self._load_and_preprocess_image(image_path)
        if image is None:
            # Fallback to a random sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        try:
            # Apply different augmentation strategies based on sample type
            if sample_type.startswith("synthetic"):
                # For synthetic samples, apply aggressive augmentation
                if hasattr(self, 'synthetic_transform'):
                    augmented = self.synthetic_transform(image=image)
                    image = augmented["image"]
                else:
                    augmented = self.transform(image=image)
                    image = augmented["image"]
            else:
                # For clean and existing distortion images, apply regular augmentation
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented["image"]

            return image, label

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

    def refresh(self):
        """Call this at the start of each epoch to resample distortions per identity."""
        self._build_dataset()

    def get_class_weights(self):
        """Calculate class weights for weighted loss"""
        class_counts = {}
        for _, label, _ in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        total_samples = len(self.samples)
        num_classes = len(class_counts)

        # Calculate weights (inverse frequency)
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            weights.append(weight)

        return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Example usage for testing
    root = "data/facecom/Task_B"
    distortions_per_identity = 3
    synthetic_per_clean = 2
    image_size = 224

    # Build identity_to_idx mapping
    identity_folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
    identity_to_idx = {name: idx for idx, name in enumerate(identity_folders)}

    # Instantiate dataset
    dataset = EnhancedFaceRecognitionDataset(
        root_dir=root,
        distortions_per_identity=distortions_per_identity,
        synthetic_per_clean=synthetic_per_clean,
        mode="train",
        image_size=image_size,
        identity_to_idx=identity_to_idx,
        balance_classes=True
    )

    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    # Try loading a sample
    img, label = dataset[0]
    print(f"Sample image shape: {img.shape}, label: {label}")

    # Check class weights
    weights = dataset.get_class_weights()
    print(f"Class weights shape: {weights.shape}, min: {weights.min():.3f}, max: {weights.max():.3f}")