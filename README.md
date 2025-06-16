# NoxVision — Gender Classification under Adverse Visual Conditions

This repository contains my solution to **Task A** of the COMSYS Hackathon-5:
**Binary Gender Classification** under blur, fog, glare, low light, and other real-world distortions using robust deep learning.

---

## Task Overview

**Task A – Gender Classification**

- **Input**: Face image (with possible visual degradation)
- **Output**: Predicted gender: `male` or `female`
- **Goal**: Train a model that is accurate, fair, and robust to common visual noise

---

## Methodology

I addressed the challenge by combining:

| Component            | Description |
|----------------------|-------------|
| **Model**            | ResNet18 backbone with a custom classification head (NoxVisionNet) |
| **Loss Function**    | Weighted Focal Loss to address gender imbalance |
| **Augmentations**    | Albumentations pipeline with simulated fog, blur, glare, occlusion |
| **Balanced Dataset** | Rebalanced training set to avoid male-only bias |
| **Evaluation**       | Accuracy, Precision, Recall, F1-score (macro) |

I focused on female recall, fairness, and model stability under distortion-heavy data.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/noxvision.git
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

To run evaluation on any test set structured as:

```
your_test_path/
├── male/
└── female/
```

Run:

```bash
python task_a_test.py --test_dir your_test_path
```

By default, the model will load `best_gender_model_balanced.pt` from root.

---

## Training

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

---

## Sample Test Output

```bash
Accuracy : 0.9052
Precision: 0.6860
Recall   : 0.9114
F1-score : 0.7834
```

This demonstrates strong recall and fairness, especially under adverse visual conditions.

---

## Submission-Ready

- Pretrained model included
- Scripts run without external config or cloud
- No GPU dependency (runs on MPS / CPU)
- No hardcoded paths
- Reproducible results

---

## Contact

If you have any issues running the code, feel free to open an issue or contact