import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from collections import defaultdict
import itertools

from utils.augmentations import get_balanced_train_transforms, get_aggressive_train_transforms, get_val_transforms

class FaceVerificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train",
                 pairs_per_identity=50, hard_negative_ratio=0.3,
                 image_size=224, identity_to_idx=None,
                 synthetic_per_clean=5):
        """
        Face Verification Dataset for Triplet/Siamese Learning

        Args:
            root_dir (str): Path to Task_B directory containing identity folders
            transform: Albumentations transform to apply
            mode (str): 'train' or 'val'
            pairs_per_identity (int): Number of pairs to generate per identity per epoch
            hard_negative_ratio (float): Ratio of hard negatives to sample
            image_size (int): Target image size
            identity_to_idx (dict): Mapping from identity names to indices
            synthetic_per_clean (int): Synthetic distortions per clean image
        """
        assert mode in {"train", "val"}, "Mode must be 'train' or 'val'"
        self.root_dir = root_dir
        self.mode = mode
        self.pairs_per_identity = pairs_per_identity
        self.hard_negative_ratio = hard_negative_ratio
        self.image_size = image_size
        self.synthetic_per_clean = synthetic_per_clean

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

        # Data structures
        self.identity_to_idx = identity_to_idx or {}
        self.identity_images = defaultdict(list)  # identity_idx -> list of image paths
        self.clean_images_per_identity = defaultdict(list)  # For synthetic generation
        self.all_identities = []
        self.pairs = []  # Will store (img1_path, img2_path, label, sample_types)

        self._build_dataset()
        self._generate_pairs()

    def _build_dataset(self):
        """Build the dataset by collecting all images per identity."""
        data_path = os.path.join(self.root_dir, self.mode)
        identity_folders = sorted(os.listdir(data_path))


        # Build identity mapping if not provided
        if not self.identity_to_idx:
            self.identity_to_idx = {
                name: idx for idx, name in enumerate(identity_folders)
                if os.path.isdir(os.path.join(data_path, name))
            }

        self.identity_images.clear()
        self.clean_images_per_identity.clear()

        for identity_name, idx in self.identity_to_idx.items():
            identity_path = os.path.join(data_path, identity_name)
            if not os.path.isdir(identity_path):
                continue

            # Collect clean images (not in distortion folder)
            clean_images = []
            for file in os.listdir(identity_path):
                file_path = os.path.join(identity_path, file)
                if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
                    clean_images.append((file_path, "clean"))

            self.clean_images_per_identity[idx] = [path for path, _ in clean_images]
            self.identity_images[idx].extend(clean_images)

            # Add synthetic distortions from clean images (training only)
            if self.mode == "train" and clean_images:
                for clean_path, _ in clean_images:
                    for i in range(self.synthetic_per_clean):
                        self.identity_images[idx].append((clean_path, f"synthetic_{i}"))

            # Collect existing distorted images
            distortion_dir = os.path.join(identity_path, "distortion")
            if os.path.isdir(distortion_dir):
                distortions = [f for f in os.listdir(distortion_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                if len(distortions) >= 2:
                    random.shuffle(distortions)
                    midpoint = len(distortions) // 2

                    # Split distortions between train/val
                    if self.mode == "train":
                        selected = distortions[:midpoint]
                    else:
                        selected = distortions[midpoint:]

                    for file in selected:
                        path = os.path.join(distortion_dir, file)
                        self.identity_images[idx].append((path, "distortion"))

        # Filter out identities with insufficient images
        min_images = 2  # Need at least 2 images to form positive pairs
        self.all_identities = [idx for idx, images in self.identity_images.items()
                              if len(images) >= min_images]

        print(f"[{self.mode.upper()}] ✅ Loaded {len(self.all_identities)} identities")
        for idx in self.all_identities[:5]:  # Show first 5
            identity_name = [k for k, v in self.identity_to_idx.items() if v == idx][0]
            print(f"  Identity {identity_name}: {len(self.identity_images[idx])} images")

    def _generate_pairs(self):
        """Generate positive and negative pairs for training/validation."""
        self.pairs.clear()

        if len(self.all_identities) < 2:
            print(f"⚠️ Not enough identities ({len(self.all_identities)}) to generate pairs")
            return

        # Generate positive pairs (same identity)
        positive_pairs = []
        for identity_idx in self.all_identities:
            images = self.identity_images[identity_idx]
            if len(images) < 2:
                continue

            # Generate all possible positive pairs for this identity
            available_pairs = list(itertools.combinations(images, 2))

            # Sample pairs for this identity
            num_pairs = min(self.pairs_per_identity, len(available_pairs))
            sampled_pairs = random.sample(available_pairs, num_pairs)

            for (img1_path, type1), (img2_path, type2) in sampled_pairs:
                positive_pairs.append((img1_path, img2_path, 1, f"{type1}_{type2}"))

        # Generate negative pairs (different identities)
        num_negatives = len(positive_pairs)  # Balanced positive/negative ratio
        negative_pairs = []

        for _ in range(num_negatives):
            # Sample two different identities
            if len(self.all_identities) < 2:
                break

            id1, id2 = random.sample(self.all_identities, 2)

            # Sample one image from each identity
            img1_path, type1 = random.choice(self.identity_images[id1])
            img2_path, type2 = random.choice(self.identity_images[id2])

            negative_pairs.append((img1_path, img2_path, 0, f"{type1}_{type2}"))

        # Combine and shuffle pairs
        self.pairs = positive_pairs + negative_pairs
        random.shuffle(self.pairs)

        # Statistics
        pos_count = sum(1 for _, _, label, _ in self.pairs if label == 1)
        neg_count = len(self.pairs) - pos_count

        print(f"[{self.mode.upper()}] 📊 Generated {len(self.pairs)} pairs:")
        print(f"  Positive pairs: {pos_count}")
        print(f"  Negative pairs: {neg_count}")
        print(f"  Balance ratio: {pos_count/max(neg_count, 1):.2f}")

    def _load_and_preprocess_image(self, image_path):
        """Load image and apply basic preprocessing."""
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
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label, sample_types = self.pairs[idx]

        # Load images
        img1 = self._load_and_preprocess_image(img1_path)
        img2 = self._load_and_preprocess_image(img2_path)

        if img1 is None or img2 is None:
            # Fallback to a random pair
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

        try:
            # Apply transforms based on sample type
            type1, type2 = sample_types.split('_')

            # Transform first image
            if type1.startswith("synthetic"):
                if hasattr(self, 'synthetic_transform'):
                    augmented1 = self.synthetic_transform(image=img1)
                    img1 = augmented1["image"]
                else:
                    augmented1 = self.transform(image=img1)
                    img1 = augmented1["image"]
            else:
                if self.transform:
                    augmented1 = self.transform(image=img1)
                    img1 = augmented1["image"]

            # Transform second image
            if type2.startswith("synthetic"):
                if hasattr(self, 'synthetic_transform'):
                    augmented2 = self.synthetic_transform(image=img2)
                    img2 = augmented2["image"]
                else:
                    augmented2 = self.transform(image=img2)
                    img2 = augmented2["image"]
            else:
                if self.transform:
                    augmented2 = self.transform(image=img2)
                    img2 = augmented2["image"]

            return img1, img2, torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            print(f"⚠️ Error processing pair ({img1_path}, {img2_path}): {e}")
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

    def refresh(self):
        """Refresh pairs - call at the start of each epoch."""
        self._generate_pairs()

    def get_identity_galleries(self):
        """Get gallery images for each identity (for evaluation)."""
        galleries = {}
        for identity_idx in self.all_identities:
            # Use clean images as gallery references
            clean_images = [(path, sample_type) for path, sample_type in self.identity_images[identity_idx]
                           if sample_type == "clean"]
            if clean_images:
                galleries[identity_idx] = clean_images
        return galleries

# Triplet Dataset for more advanced triplet learning
class FaceTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train",
                 triplets_per_identity=30, image_size=224,
                 identity_to_idx=None, synthetic_per_clean=5):
        """
        Face Triplet Dataset for advanced triplet learning

        Args:
            triplets_per_identity (int): Number of triplets per identity per epoch
            Other args same as FaceVerificationDataset
        """
        self.root_dir = root_dir
        self.mode = mode
        self.triplets_per_identity = triplets_per_identity
        self.image_size = image_size
        self.synthetic_per_clean = synthetic_per_clean

        # Initialize same as verification dataset
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

        self.identity_to_idx = identity_to_idx or {}
        self.identity_images = defaultdict(list)
        self.clean_images_per_identity = defaultdict(list)
        self.all_identities = []
        self.triplets = []  # (anchor_path, positive_path, negative_path, types)

        self._build_dataset()
        self._generate_triplets()

    def _build_dataset(self):
        """Same as FaceVerificationDataset._build_dataset()"""
        data_path = os.path.join(self.root_dir, self.mode)
        identity_folders = sorted(os.listdir(data_path))


        if not self.identity_to_idx:
            self.identity_to_idx = {
                name: idx for idx, name in enumerate(identity_folders)
                if os.path.isdir(os.path.join(data_path, name))
            }


        self.identity_images.clear()
        self.clean_images_per_identity.clear()

        for identity_name, idx in self.identity_to_idx.items():
            identity_path = os.path.join(data_path, identity_name)
            if not os.path.isdir(identity_path):
                continue

            # Collect clean images
            clean_images = []
            for file in os.listdir(identity_path):
                file_path = os.path.join(identity_path, file)
                if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
                    clean_images.append((file_path, "clean"))

            self.clean_images_per_identity[idx] = [path for path, _ in clean_images]
            self.identity_images[idx].extend(clean_images)

            # Add synthetic distortions (training only)
            if self.mode == "train" and clean_images:
                for clean_path, _ in clean_images:
                    for i in range(self.synthetic_per_clean):
                        self.identity_images[idx].append((clean_path, f"synthetic_{i}"))

            # Collect existing distorted images
            distortion_dir = os.path.join(identity_path, "distortion")
            if os.path.isdir(distortion_dir):
                distortions = [f for f in os.listdir(distortion_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                if len(distortions) >= 2:
                    random.shuffle(distortions)
                    midpoint = len(distortions) // 2

                    if self.mode == "train":
                        selected = distortions[:midpoint]
                    else:
                        selected = distortions[midpoint:]

                    for file in selected:
                        path = os.path.join(distortion_dir, file)
                        self.identity_images[idx].append((path, "distortion"))

        # Filter identities with sufficient images
        min_images = 2
        self.all_identities = [idx for idx, images in self.identity_images.items()
                              if len(images) >= min_images]

        print(f"[{self.mode.upper()} TRIPLET] ✅ Loaded {len(self.all_identities)} identities")

    def _generate_triplets(self):
        """Generate triplets: (anchor, positive, negative)"""
        self.triplets.clear()

        if len(self.all_identities) < 2:
            print(f"⚠️ Not enough identities for triplets")
            return

        for anchor_identity in self.all_identities:
            anchor_images = self.identity_images[anchor_identity]
            if len(anchor_images) < 2:
                continue

            # Generate triplets for this identity
            for _ in range(self.triplets_per_identity):
                # Sample anchor and positive from same identity
                anchor_img, anchor_type = random.choice(anchor_images)
                positive_candidates = [img for img in anchor_images if img[0] != anchor_img]
                if not positive_candidates:
                    continue
                positive_img, positive_type = random.choice(positive_candidates)

                # Sample negative from different identity
                negative_identities = [idx for idx in self.all_identities if idx != anchor_identity]
                if not negative_identities:
                    continue
                negative_identity = random.choice(negative_identities)
                negative_img, negative_type = random.choice(self.identity_images[negative_identity])

                triplet = (
                    anchor_img, positive_img, negative_img,
                    f"{anchor_type}|{positive_type}|{negative_type}"
                )
                self.triplets.append(triplet)

        random.shuffle(self.triplets)
        print(f"[{self.mode.upper()} TRIPLET] 📊 Generated {len(self.triplets)} triplets")

    def _load_and_preprocess_image(self, image_path):
        """Same as FaceVerificationDataset._load_and_preprocess_image()"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

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
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path, sample_types = self.triplets[idx]

        # Load images
        anchor = self._load_and_preprocess_image(anchor_path)
        positive = self._load_and_preprocess_image(positive_path)
        negative = self._load_and_preprocess_image(negative_path)

        if anchor is None or positive is None or negative is None:
            return self.__getitem__(random.randint(0, len(self.triplets) - 1))

        try:
            anchor_type, positive_type, negative_type = sample_types.split('|')

            # Apply transforms
            def apply_transform(img, img_type):
                if img_type.startswith("synthetic"):
                    transform = getattr(self, 'synthetic_transform', self.transform)
                else:
                    transform = self.transform

                if transform:
                    augmented = transform(image=img)
                    return augmented["image"]
                return torch.from_numpy(img).permute(2, 0, 1).float()

            anchor = apply_transform(anchor, anchor_type)
            positive = apply_transform(positive, positive_type)
            negative = apply_transform(negative, negative_type)

            return anchor, positive, negative

        except Exception as e:
            print(f"⚠️ Error processing triplet: {e}")
            return self.__getitem__(random.randint(0, len(self.triplets) - 1))

    def refresh(self):
        """Refresh triplets - call at the start of each epoch."""
        self._generate_triplets()


if __name__ == "__main__":
    # Test the datasets
    root = "data/facecom/Task_B"

    # Build identity mapping
    identity_folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
    identity_to_idx = {name: idx for idx, name in enumerate(identity_folders)}

    print("Testing FaceVerificationDataset...")
    verification_dataset = FaceVerificationDataset(
        root_dir=root,
        mode="train",
        pairs_per_identity=20,
        image_size=224,
        identity_to_idx=identity_to_idx,
        synthetic_per_clean=3
    )

    print(f"Verification dataset size: {len(verification_dataset)}")
    if len(verification_dataset) > 0:
        img1, img2, label = verification_dataset[0]
        print(f"Sample: img1 shape: {img1.shape}, img2 shape: {img2.shape}, label: {label}")

    print("\nTesting FaceTripletDataset...")
    triplet_dataset = FaceTripletDataset(
        root_dir=root,
        mode="train",
        triplets_per_identity=15,
        image_size=224,
        identity_to_idx=identity_to_idx,
        synthetic_per_clean=3
    )

    print(f"Triplet dataset size: {len(triplet_dataset)}")
    if len(triplet_dataset) > 0:
        anchor, positive, negative = triplet_dataset[0]
        print(f"Sample: anchor: {anchor.shape}, positive: {positive.shape}, negative: {negative.shape}")