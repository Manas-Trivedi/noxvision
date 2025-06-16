import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import collections
from collections import Counter
from models.model import NoxVisionNet
from utils.gender_dataset import GenderDataset

import os

def main():
    # 🔧 Configs
    train_path = "data/facecom/task_a/train"
    val_path = "data/facecom/task_a/val"
    batch_size = 32
    epochs = 10
    lr = 1e-3
    num_workers = 4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    save_path = "best_gender_model.pt"

    # 📦 Datasets
    train_dataset = GenderDataset(train_path, is_train=True)
    val_dataset = GenderDataset(val_path, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 🧠 Model
    model = NoxVisionNet(num_classes_identity=1).to(device)  # dummy for identity
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            gender_logits, _ = model(images)
            loss = loss_fn(gender_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ✅ Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                gender_logits, _ = model(images)
                preds = torch.argmax(gender_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

        # 💾 Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model to {save_path}")

if __name__ == "__main__":
    main()
