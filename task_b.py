import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict, Counter

from models.face_model import FaceNet  # ✅ using your actual model

# ✅ Match train-time preprocessing
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_face_data(folder):
    """Load face images organized by identity folders"""
    images = []
    identities = []

    identity_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    for identity_name in identity_folders:
        identity_dir = os.path.join(folder, identity_name)
        for fname in os.listdir(identity_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(identity_dir, fname)
                images.append(path)
                identities.append(identity_name)

    print(f"📊 Loaded {len(images)} images from {len(set(identities))} identities")
    return images, identities

def create_verification_pairs(images, identities, num_pairs=1000):
    """Create positive and negative face verification pairs"""
    identity_to_images = defaultdict(list)
    for img_path, identity in zip(images, identities):
        identity_to_images[identity].append(img_path)

    # Filter identities with at least 2 images
    valid_identities = {identity: imgs for identity, imgs in identity_to_images.items() if len(imgs) >= 2}

    if len(valid_identities) == 0:
        raise ValueError("No identities with multiple images found for verification pairs")

    positive_pairs = []
    negative_pairs = []

    # Create positive pairs (same identity)
    identity_list = list(valid_identities.keys())
    pairs_per_identity = max(1, num_pairs // (2 * len(identity_list)))

    for identity, imgs in valid_identities.items():
        for _ in range(min(pairs_per_identity, len(imgs)//2)):
            if len(imgs) >= 2:
                img1, img2 = random.sample(imgs, 2)
                positive_pairs.append((img1, img2, 1))  # Label 1 for same person

    # Create negative pairs (different identities)
    for _ in range(len(positive_pairs)):
        if len(identity_list) >= 2:
            id1, id2 = random.sample(identity_list, 2)
            img1 = random.choice(valid_identities[id1])
            img2 = random.choice(valid_identities[id2])
            negative_pairs.append((img1, img2, 0))  # Label 0 for different person

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    print(f"🔄 Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
    return all_pairs

def compute_embedding(model, img_path, device):
    """Compute face embedding for a single image"""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
        # L2 normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

def main():
    parser = argparse.ArgumentParser(description="Face Verification Inference (Task B)")
    parser.add_argument('--test_dir', type=str, required=True,
                       help="Path to Task_B test folder (with identity subfolders)")
    parser.add_argument('--weights', type=str, default='face_model.pt',
                       help="Path to pretrained face model weights")
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help="Embedding dimension (must match training)")
    parser.add_argument('--num_pairs', type=int, default=1000,
                       help="Number of verification pairs to create")
    parser.add_argument('--threshold', type=float, default=None,
                       help="Cosine similarity threshold (auto-determined if not provided)")
    args = parser.parse_args()

    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # Load model
    print(f"📥 Loading model with embedding dimension: {args.embedding_dim}")
    model = FaceNet(embedding_dim=args.embedding_dim).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load test data
    print(f"📂 Loading test data from: {args.test_dir}")
    img_paths, identities = load_face_data(args.test_dir)

    # Create verification pairs
    print("🔄 Creating verification pairs...")
    pairs = create_verification_pairs(img_paths, identities, args.num_pairs)

    if len(pairs) == 0:
        print("❌ No verification pairs could be created!")
        return

    # Run inference
    print("🔍 Computing face embeddings and similarities...")
    similarities = []
    labels = []

    for img1_path, img2_path, label in tqdm(pairs, desc="Processing pairs"):
        try:
            # Get embeddings
            emb1 = compute_embedding(model, img1_path, device)
            emb2 = compute_embedding(model, img2_path, device)

            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
            similarities.append(similarity)
            labels.append(label)

        except Exception as e:
            print(f"⚠️  Error processing pair ({img1_path}, {img2_path}): {e}")
            continue

    if len(similarities) == 0:
        print("❌ No valid pairs could be processed!")
        return

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find optimal threshold if not provided
    if args.threshold is None:
        print("🎯 Finding optimal threshold...")
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)
        best_acc = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            predictions = (similarities > threshold).astype(int)
            acc = accuracy_score(labels, predictions)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        optimal_threshold = best_threshold
        print(f"📊 Optimal threshold found: {optimal_threshold:.4f}")
    else:
        optimal_threshold = args.threshold
        print(f"🎯 Using provided threshold: {optimal_threshold:.4f}")

    # Calculate final metrics
    final_predictions = (similarities > optimal_threshold).astype(int)

    accuracy = accuracy_score(labels, final_predictions)
    precision = precision_score(labels, final_predictions, zero_division=1)
    recall = recall_score(labels, final_predictions, zero_division=1)
    f1 = f1_score(labels, final_predictions, zero_division=1)

    # Calculate AUC
    try:
        auc = roc_auc_score(labels, similarities)
    except:
        auc = 0.0

    # Print results
    print("\n" + "="*50)
    print("📊 FACE VERIFICATION RESULTS (Task B)")
    print("="*50)
    print(f"Threshold      : {optimal_threshold:.4f}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"AUC            : {auc:.4f}")
    print(f"Total pairs    : {len(pairs)}")
    print(f"Processed pairs: {len(similarities)}")
    print("="*50)

    # Additional statistics
    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]

    print(f"\n📈 Similarity Statistics:")
    print(f"Positive pairs (same person) - Mean: {pos_sims.mean():.4f}, Std: {pos_sims.std():.4f}")
    print(f"Negative pairs (diff person) - Mean: {neg_sims.mean():.4f}, Std: {neg_sims.std():.4f}")

    print(f"\n🎯 Prediction Breakdown:")
    print(f"Predicted Same Person    : {Counter(final_predictions)[1]}")
    print(f"Predicted Different Person: {Counter(final_predictions)[0]}")
    print(f"True Same Person         : {Counter(labels)[1]}")
    print(f"True Different Person    : {Counter(labels)[0]}")

if __name__ == "__main__":
    main()