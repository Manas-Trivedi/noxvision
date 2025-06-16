import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm
from collections import Counter

from models.model import NoxVisionNet  # ✅ using your actual model

# ✅ Match train-time preprocessing
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def load_data(folder):
    images, labels = [], []
    label_map = {'male': 0, 'female': 1}
    for label_name in ['male', 'female']:
        class_dir = os.path.join(folder, label_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Expected subfolder {class_dir} not found.")
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(class_dir, fname)
                images.append(path)
                labels.append(label_map[label_name])
    return images, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help="Path to Task_A test folder (with male/ and female/)")
    parser.add_argument('--weights', type=str, default='gender_model.pt', help="Path to pretrained weights")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    model = NoxVisionNet(num_classes_identity=1).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img_paths, true_labels = load_data(args.test_dir)
    all_preds = []

    for path in tqdm(img_paths, desc="🔍 Running Inference"):
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device) # type: ignore
        with torch.no_grad():
            gender_logits, _ = model(tensor)
            pred = torch.argmax(gender_logits, dim=1).item()
            all_preds.append(pred)

    acc = accuracy_score(true_labels, all_preds)
    precision = precision_score(true_labels, all_preds, zero_division=1)
    recall = recall_score(true_labels, all_preds, zero_division=1)
    f1 = f1_score(true_labels, all_preds, zero_division=1)

    print("\n📊 Test Results (Task A):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Prediction Breakdown:", Counter(all_preds))
    print("True Label Breakdown:", Counter(true_labels))

if __name__ == "__main__":
    main()
