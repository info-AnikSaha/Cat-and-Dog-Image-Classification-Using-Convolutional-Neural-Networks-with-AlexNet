# 🐱🐶 Cat vs Dog Image Classification using AlexNet (Transfer Learning)

A binary image classifier that distinguishes **cats** from **dogs** using a **pretrained AlexNet** CNN fine-tuned with PyTorch. The model reaches **96.33% test accuracy** and a **ROC AUC of 0.9949** on the Microsoft Cats vs Dogs dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.33%25-brightgreen)

> Course project — Artificial Intelligence, Department of CSE, East West University.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Model & Training Configuration](#-model--training-configuration)
- [Results](#-results)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🔎 Overview

Given an input image, the model predicts whether it shows a **Cat** or a **Dog**. Instead of training from scratch, the project uses **transfer learning**: the convolutional feature extractor of a pretrained AlexNet is frozen, and only the classifier head is trained, with the final layer replaced by a 2-class output.

**Highlights**

- Dataset inspection, corrupted-image check and blur check
- Data augmentation (rotation, flip, zoom/shift, brightness)
- Reproducible 70 / 15 / 15 train–validation–test split (seed = 42)
- Transfer learning with pretrained AlexNet (frozen feature extractor)
- Full evaluation: accuracy, precision, recall, F1, confusion matrix, ROC and Precision–Recall curves

---

## 📂 Dataset

**Microsoft Cats vs Dogs** dataset, downloaded automatically through KaggleHub (`shaunthesheep/microsoft-catsvsdogs-dataset`).

| Item | Value |
| --- | --- |
| Classes | 2 (Cat = 0, Dog = 1) |
| Cat images | 12,176 (49.61%) |
| Dog images | 12,366 (50.39%) |
| **Total images** | **24,542** |
| Corrupted images | 0 |
| Severely blurred images | 0 |
| Train / Val / Test | 17,179 / 3,681 / 3,682 |

The dataset is nearly balanced, so there is no strong class-imbalance bias.

---

## ⚙️ Pipeline

1. **Download** the dataset with KaggleHub
2. **Inspect** class distribution (bar & pie charts) and sample images
3. **Verify** data quality — corrupted-image check and Laplacian-variance blur check
4. **Preprocess**
   - Resize to `224 × 224`
   - Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
   - Training-only augmentation: random rotation (±20°), horizontal flip, affine translate/zoom, brightness jitter
5. **Split** into train / validation / test (70 / 15 / 15)
6. **Train** the AlexNet classifier head for 10 epochs
7. **Evaluate** on the held-out test set

---

## 🧠 Model & Training Configuration

| Setting | Value |
| --- | --- |
| Base model | Pretrained AlexNet (`torchvision`, default weights) |
| Feature extractor | Frozen |
| Final layer | `Linear(in_features, 2)` |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam (classifier parameters only) |
| Learning rate | `0.0001` |
| Epochs | 10 |
| Batch size | 128 |
| Hardware | NVIDIA GeForce RTX 3050 (CUDA 12.4) |
| PyTorch | 2.6.0+cu124 |

---

## 📊 Results

### Test set performance

| Metric | Value |
| --- | --- |
| **Accuracy** | **96.33%** |
| Precision | 96.88% |
| Recall | 96.02% |
| F1-Score | 96.45% |
| Cat AUC | 0.9949 |
| Dog AUC | 0.9949 |

### Confusion matrix (3,682 test images)

| True \ Predicted | Cat | Dog |
| --- | --- | --- |
| **Cat** | 1,714 | 59 |
| **Dog** | 76 | 1,833 |

### Training history

| Epoch | Train Loss | Train Acc | Val Acc |
| --- | --- | --- | --- |
| 1 | 0.1854 | 92.20% | 96.01% |
| 2 | 0.1372 | 94.31% | 95.52% |
| 3 | 0.1230 | 94.96% | 96.25% |
| 4 | 0.1116 | 95.43% | **96.63%** |
| 5 | 0.1050 | 95.65% | 96.50% |
| 6 | 0.0969 | 96.13% | 96.60% |
| 7 | 0.0973 | 96.09% | 96.25% |
| 8 | 0.0837 | 96.64% | 96.44% |
| 9 | 0.0884 | 96.51% | 96.22% |
| 10 | 0.0792 | 96.82% | 96.22% |

Training and validation accuracy stay close together (96.82% vs 96.22% at epoch 10), so no severe overfitting is observed.

> 💡 Tip: add screenshots of your confusion matrix, ROC curve and augmentation examples in an `images/` folder and embed them here, e.g. `![Confusion Matrix](images/confusion_matrix.png)`.

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/mohammednaeemog-sys/-Image-Classification-Using-Convolutional-Neural-Network.git
cd -Image-Classification-Using-Convolutional-Neural-Network
```

### 2. Install dependencies

```bash
pip install torch torchvision kagglehub opencv-python pillow matplotlib scikit-learn jupyter
```

> For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Run the notebook

```bash
jupyter notebook
```

Open the notebook and run all cells in order. The dataset is downloaded automatically via KaggleHub (you may need to configure Kaggle credentials on first use).

---

## 📁 Project Structure

```
.
├── Cat-Dog-Classifier-AlexNet.ipynb          # Full pipeline: data, training, evaluation
├── Cat_Dog_AlexNet_CNN_Project_Report.docx   # Detailed project report
└── README.md
```

---

## 🔮 Future Work

- Systematic hyperparameter tuning (learning rate, batch size, epochs)
- Fine-tune selected convolutional layers instead of freezing the whole feature extractor
- Add regularization (dropout, weight decay)
- Compare with VGG, ResNet and MobileNet
- Test on real-world images outside the dataset
- Deploy as a simple web / mobile app for interactive prediction

---

## 📚 References

1. Project repository — [Image Classification Using Convolutional Neural Network](https://github.com/mohammednaeemog-sys/-Image-Classification-Using-Convolutional-Neural-Network)
2. Microsoft Cats vs Dogs dataset — Kaggle: `shaunthesheep/microsoft-catsvsdogs-dataset`
3. Krizhevsky, A., Sutskever, I., Hinton, G. — *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet), 2012

---

## 👤 Author

**Anik Saha** — CSE, East West University

---

## 📄 License

This project is for educational purposes.
