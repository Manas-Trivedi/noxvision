import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from utils.augmentations import get_val_transforms

class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else get_val_transforms()
        self.samples = []

        for identity in os.listdir(root_dir):
            identity_path = os.path.join(root_dir, identity)
            if not os.path.isdir(identity_path):
                continue

            # Add the clean image
            for file in os.listdir(identity_path):
                if file.endswith(".jpg") and not file.startswith("."):
                    if file != "distortion":
                        image_path = os.path.join(identity_path, file)
                        self.samples.append((image_path, identity))

            # Add distorted versions
            distortion_path = os.path.join(identity_path, "distortion")
            if os.path.exists(distortion_path):
                for dist_file in os.listdir(distortion_path):
                    if dist_file.endswith(".jpg"):
                        image_path = os.path.join(distortion_path, dist_file)
                        self.samples.append((image_path, identity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, identity = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, identity 
