# utils/gender_dataset_v2.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from collections import Counter
import random
from utils.augmentations import (
    get_balanced_train_transforms,
    get_aggressive_train_transforms,
    get_val_transforms
)

class BalancedGenderDataset(Dataset):
    def __init__(self, root_dir, image_size=224, is_train=True, boost_minority=True):
        """
        root_dir: path to train/ or val/
        boost_minority: Apply more aggressive augmentation to minority class
        """
        self.image_paths = []
        self.labels = []
        self.is_train = is_train
        self.boost_minority = boost_minority and is_train

        # Load data
        for label_str, label_int in [("male", 0), ("female", 1)]:
            folder = os.path.join(root_dir, label_str)
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist!")
                continue

            image_files = glob(os.path.join(folder, "*.jpg")) + \
                         glob(os.path.join(folder, "*.png")) + \
                         glob(os.path.join(folder, "*.jpeg"))

            self.image_paths.extend(image_files)
            self.labels.extend([label_int] * len(image_files))

        # Calculate class distribution
        self.class_counts = Counter(self.labels)
        self.minority_class = min(self.class_counts, key=lambda k: self.class_counts[k])

        print(f"Dataset loaded: {dict(self.class_counts)}")
        print(f"Minority class: {self.minority_class} ({'female' if self.minority_class == 1 else 'male'})")

        # Setup transforms
        if is_train:
            self.normal_transform = get_balanced_train_transforms(image_size)
            if self.boost_minority:
                self.aggressive_transform = get_aggressive_train_transforms(image_size)
        else:
            self.normal_transform = get_val_transforms(image_size)
            self.aggressive_transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not read image {path}")
            # Return a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Choose augmentation strategy
        if self.boost_minority and label == self.minority_class:
            # Apply more aggressive augmentation to minority class
            if random.random() < 0.7:  # 70% chance for aggressive augmentation
                transform = self.aggressive_transform
            else:
                transform = self.normal_transform
        else:
            transform = self.normal_transform

        # Apply transformation
        if transform:
            try:
                image = transform(image=image)["image"]
            except Exception as e:
                print(f"Transform error for {path}: {e}")
                # Fallback to basic transform
                basic_transform = get_val_transforms()
                image = basic_transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        """Calculate class weights for loss function"""
        total = len(self.labels)
        weights = []
        for i in range(len(self.class_counts)):
            weight = total / (len(self.class_counts) * self.class_counts[i])
            weights.append(weight)
        return torch.FloatTensor(weights)

# Alternative: Oversampling dataset that duplicates minority samples
class OversampledGenderDataset(Dataset):
    def __init__(self, root_dir, image_size=224, is_train=True, oversample_ratio=1.0):
        """
        oversample_ratio: How much to oversample minority class (1.0 = balance classes)
        """
        self.image_size = image_size
        self.is_train = is_train

        # Load original data
        male_paths = []
        female_paths = []

        male_folder = os.path.join(root_dir, "male")
        female_folder = os.path.join(root_dir, "female")

        if os.path.exists(male_folder):
            male_paths = glob(os.path.join(male_folder, "*.jpg")) + \
                        glob(os.path.join(male_folder, "*.png")) + \
                        glob(os.path.join(male_folder, "*.jpeg"))

        if os.path.exists(female_folder):
            female_paths = glob(os.path.join(female_folder, "*.jpg")) + \
                          glob(os.path.join(female_folder, "*.png")) + \
                          glob(os.path.join(female_folder, "*.jpeg"))

        print(f"Original counts - Male: {len(male_paths)}, Female: {len(female_paths)}")

        # Determine oversampling
        if len(male_paths) > len(female_paths):
            # Oversample females
            majority_count = len(male_paths)
            minority_paths = female_paths
            minority_label = 1

            target_minority_count = int(majority_count * oversample_ratio)

            # Repeat minority samples
            oversampled_minority = []
            while len(oversampled_minority) < target_minority_count:
                oversampled_minority.extend(minority_paths)
            oversampled_minority = oversampled_minority[:target_minority_count]

            self.image_paths = male_paths + oversampled_minority
            self.labels = [0] * len(male_paths) + [1] * len(oversampled_minority)

        else:
            # Oversample males (unlikely given your data)
            majority_count = len(female_paths)
            minority_paths = male_paths
            minority_label = 0

            target_minority_count = int(majority_count * oversample_ratio)

            oversampled_minority = []
            while len(oversampled_minority) < target_minority_count:
                oversampled_minority.extend(minority_paths)
            oversampled_minority = oversampled_minority[:target_minority_count]

            self.image_paths = female_paths + oversampled_minority
            self.labels = [1] * len(female_paths) + [0] * len(oversampled_minority)

        print(f"After oversampling: {Counter(self.labels)}")

        # Setup transforms
        if is_train:
            self.transform = get_balanced_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            try:
                image = self.transform(image=image)["image"]
            except:
                basic_transform = get_val_transforms(self.image_size)
                image = basic_transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)