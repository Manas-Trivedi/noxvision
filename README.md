# NoxVision — Gender Classification and Face Verification under Adverse Visual Conditions

This repository contains my solution to the COMSYS Hackathon-5:
**Binary Gender Classification** and **Face Verification** under blur, fog, glare, low light, and other real-world distortions using robust deep learning.

---

## Task Overview

### Task A – Gender Classification

- **Input**: Face image (with possible visual degradation)
- **Output**: Predicted gender: `male` or `female`
- **Goal**: Train a model that is accurate, fair, and robust to common visual noise

### Task B – Face Verification

- **Input**: Two face images (may be degraded)
- **Output**: Whether the two images are of the same person (binary classification)
- **Goal**: Build a model that produces robust face embeddings for reliable verification under distortions

---

## Methodology

### Task A

| Component            | Description |
|----------------------|-------------|
| **Model**            | ResNet18 backbone with a custom classification head (NoxVisionNet) |
| **Loss Function**    | Weighted Focal Loss to address gender imbalance |
| **Augmentations**    | Albumentations pipeline with simulated fog, blur, glare, occlusion |
| **Balanced Dataset** | Rebalanced training set to avoid male-only bias |
| **Evaluation**       | Accuracy, Precision, Recall, F1-score (macro) |

I focused on female recall, fairness, and model stability under distortion-heavy data.

### Task B

| Component            | Description |
|----------------------|-------------|
| **Model**            | ResNet50 backbone (FaceNet) with a custom embedding head |
| **Loss Function**    | Online Triplet Loss for learning discriminative embeddings |
| **Augmentations**    | Aggressive augmentations for robustness (blur, fog, occlusion, etc.) |
| **Verification**     | Cosine similarity between L2-normalized embeddings |
| **Thresholding**     | Optimal threshold selection on validation data |
| **Evaluation**       | Accuracy, F1-score, ROC-AUC on positive/negative verification pairs |

I used hard triplet mining within each batch and evaluated with both positive (same identity) and negative (different identity) pairs, ensuring the model is robust to distortions.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Manas_Trivedi/noxvision.git
   cd noxvision
   ```

2. **Create & activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Testing

### Task A (Gender Classification)

To run evaluation on any test set structured as:

```
your_test_path/
├── male/
└── female/
```

Run:

```bash
python task_a.py --test_dir your_test_path
```

By default, the model will load `gender_model.pt` from root.

### Task B (Face Verification)

To run face verification on a test set structured as:

```
your_test_path/
├── person1/
│   ├── img1.jpg
│   └── ...
├── person2/
│   ├── img1.jpg
│   └── ...
└── ...
```

Run:

```bash
python task_b.py --test_dir your_test_path
```

You can specify the number of verification pairs, embedding dimension, and threshold if needed. By default, the model will load `face_model.pt` from root.

---

## Training

### Task A

To train from scratch on the FACECOM dataset:

```bash
python train.py
```

- Make sure the training data is placed at:
  ```
  data/facecom/Task_A/train/
    ├── male/
    └── female/
  data/facecom/Task_A/val/
    ├── male/
    └── female/
  ```

- The best model is automatically saved as:
  ```
  best_gender_model_balanced.pt
  ```

### Task B

To train the face verification model with triplet loss:

```bash
python train_b.py
```

- Make sure the training and validation data are placed at:
  ```
  data/facecom/Task_B/train/
    ├── person1/
    ├── person2/
    └── ...
  data/facecom/Task_B/val/
    ├── person1/
    ├── person2/
    └── ...
  ```

- The best model is automatically saved as:
  ```
  face_model.pt
  ```

---

## Sample Test Output

### Task A

```bash
Accuracy : 0.9052
Precision: 0.6860
Recall   : 0.9114
F1-score : 0.8034
```

### Task B

```bash
Threshold      : 0.5123
Accuracy       : 0.8720
Precision      : 0.8701
Recall         : 0.8740
F1-score       : 0.8720
AUC            : 0.9421
Total pairs    : 2000
Processed pairs: 2000
```

---

## Submission-Ready

- Pretrained models included
- Scripts run without external config or cloud
- No GPU dependency (runs on MPS / CPU)
- No hardcoded paths
- Reproducible results

---

## Contact

If you have any issues running the code, feel free to open an issue or contact