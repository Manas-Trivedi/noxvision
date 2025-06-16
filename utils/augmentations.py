# utils/augmentations.py

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

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

def simulate_occlusion(image, **kwargs):
    """
    Simulate scarf/sunglasses occlusion.
    image: np.array, shape [H, W, C]
    """
    assert isinstance(image, np.ndarray)

    h, w, _ = image.shape
    mask_type = random.choice(["scarf", "sunglasses"])

    if mask_type == "scarf":
        image[h//2:] = 0  # black out bottom half
    else:  # sunglasses
        image[h//3:h//2] = 0  # black out middle strip (eye region)

    return image

def get_train_transforms(image_size=224, apply_occlusion=True):
    return A.Compose([
        A.Lambda(image=tan_triggs_preprocessing),
        A.Lambda(image=simulate_occlusion) if apply_occlusion else A.NoOp(),
        A.RandomFog(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
