# utils/augmentations.py
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import List, Any

def tan_triggs_preprocessing(img, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1.0, sigma1=2.0, **kwargs):
    """
    Tan & Triggs normalization. Input: np.uint8 BGR image.
    """
    assert isinstance(img, np.ndarray)
    img = np.float32(img) + 1.0
    img = np.power(img, gamma)
    img = cv2.GaussianBlur(img, (0, 0), sigma0)
    img2 = cv2.GaussianBlur(img, (0, 0), sigma1)
    img = img - img2
    norm = np.power(np.abs(img), alpha)
    mean = np.power(np.mean(norm), 1.0/alpha)
    img = img / (np.power(mean, alpha) + tau)
    img = np.clip(img * 128 + 128, 0, 255).astype(np.uint8)
    return img

def clahe_equalization(img, **kwargs):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def simulate_adverse_conditions(image, **kwargs):
    """
    Simulate various adverse conditions more realistically
    """
    assert isinstance(image, np.ndarray)
    h, w, c = image.shape

    condition = random.choice([
        "motion_blur", "fog", "rain", "low_light",
        "overexposed", "uneven_lighting", "normal"
    ])

    if condition == "motion_blur":
        # Motion blur with random direction
        kernel_size = random.randint(5, 15)
        angle = random.randint(0, 180)
        kernel = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
        kernel = kernel[:2, :]
        image = cv2.filter2D(image, -1, kernel)

    elif condition == "fog":
        # Simulate fog by blending with white noise
        fog_intensity = random.uniform(0.3, 0.7)
        fog = np.ones_like(image) * 255 * fog_intensity
        image = cv2.addWeighted(image, 1-fog_intensity, fog.astype(np.uint8), fog_intensity, 0)

    elif condition == "rain":
        # Add rain-like noise
        rain_drops = np.random.random((h, w)) < 0.01
        rain_intensity = random.randint(50, 200)
        image = image.copy()
        image[rain_drops] = rain_intensity

    elif condition == "low_light":
        # Darken the image and add noise
        gamma = random.uniform(0.3, 0.7)
        image = np.power(image / 255.0, gamma) * 255
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image + noise, 0, 255)

    elif condition == "overexposed":
        # Overexpose the image
        gamma = random.uniform(1.5, 2.5)
        image = np.power(image / 255.0, gamma) * 255
        image = np.clip(image, 0, 255)

    elif condition == "uneven_lighting":
        # Create uneven lighting with gradient
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) ** 0.5
        mask = mask / mask.max()
        lighting = 1 + (mask * random.uniform(-0.5, 0.5))
        image = image * lighting[:, :, np.newaxis]
        image = np.clip(image, 0, 255)

    return image.astype(np.uint8)

def get_balanced_train_transforms(image_size=224):
    """
    Enhanced augmentation pipeline with more aggressive augmentation
    for minority class (female) samples
    """
    transforms: List[Any] = [
        # Preprocessing
        A.Lambda(image=tan_triggs_preprocessing, p=0.3),
        A.Lambda(image=clahe_equalization, p=0.3),

        # Geometric transformations
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=20,
            p=0.5
        ),

        # Adverse conditions simulation
        A.Lambda(image=simulate_adverse_conditions, p=0.6),

        # Color/lighting augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),

        # Noise and blur
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),

        A.GaussNoise(
        std_range=(0.1, 0.5),
        mean_range=(0, 0),
        per_channel=True,
        noise_scale_factor=1,
        p=0.3
        ),

        # Weather effects
        A.OneOf([
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.GaussNoise(std_range=(0.1, 0.5), mean_range=(0, 0), per_channel=True, noise_scale_factor=1, p=0.1)
    ], p=0.3),

        # Final preprocessing
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(transforms)

def get_aggressive_train_transforms(image_size=224):
    """
    More aggressive augmentation for minority class samples
    """
    transforms: List[Any] = [
        # More aggressive preprocessing
        A.Lambda(image=tan_triggs_preprocessing, p=0.5),
        A.Lambda(image=clahe_equalization, p=0.5),

        # More geometric variations
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.3,
            rotate_limit=30,
            p=0.7
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        # Stronger adverse conditions
        A.Lambda(image=simulate_adverse_conditions, p=0.8),

        # More color variations
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),

        # Stronger noise and effects
        A.OneOf([
            A.MotionBlur(blur_limit=10, p=1.0),
            A.GaussianBlur(blur_limit=10, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.6),

        A.GaussNoise(
        std_range=(0.1, 0.5),
        mean_range=(0, 0),
        per_channel=True,
        noise_scale_factor=1,
        p=0.5
        ),

        # More weather effects
        A.OneOf([
            A.GaussianBlur(blur_limit=10, p=1.0),
            A.GaussNoise(std_range=(0.1, 0.5), mean_range=(0, 0), per_channel=True, noise_scale_factor=1, p=1.0)
        ], p=0.5),

        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(transforms)

def get_val_transforms(image_size=224):
    """Minimal augmentation for validation"""
    transforms: List[Any] = [
        # In get_val_transforms
        A.Lambda(image=tan_triggs_preprocessing, p=1.0),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(transforms)