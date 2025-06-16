# utils/gender_dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
from utils.augmentations import get_train_transforms

class GenderDataset(Dataset):
    def __init__(self, root_dir, image_size=224, is_train=True):
        """
        root_dir: path to train/ or val/
        """
        self.image_paths = []
        self.labels = []
        self.transform = get_train_transforms(image_size=image_size, apply_occlusion=is_train)

        for label_str, label_int in [("male", 0), ("female", 1)]:
            folder = os.path.join(root_dir, label_str)
            image_files = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png"))
            self.image_paths.extend(image_files)
            self.labels.extend([label_int] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label)
