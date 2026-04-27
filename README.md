# 🧠 Brain Tumor Detection & Segmentation — MONAI Pipeline

This repository contains a professional-grade implementation of a brain tumor classification and segmentation pipeline using the **MONAI** framework. It is designed to handle MRI scans and classify them into four categories: Glioma, Meningioma, Pituitary, or No Tumor.

## 🚀 Features
- **MONAI-Powered:** Uses medical-specific transforms and architectures.
- **Deep Learning:** Implements `DenseNet121` with transfer learning.
- **Explainable AI:** Includes **Grad-CAM** visualizations to show model focus.
- **Kaggle Integration:** Automated dataset download and setup.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/davacodee-art/AI-engineer.git
   cd AI-engineer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Setup

This project uses the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle.

### Automated Download
To download the dataset automatically via the notebook:
1. Go to your Kaggle account settings and click **"Create New API Token"**.
2. Download the `kaggle.json` file.
3. Place `kaggle.json` in the root directory of this project.
4. The notebook will automatically configure the Kaggle API and download the data.

## 📂 Project Structure
- `tumor_monai.ipynb`: Main pipeline (Data loading -> Transforms -> Training -> Grad-CAM).
- `tumor.ipynb`: Supplementary analysis.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Configured to ignore `kaggle.json` and large data folders for security.

## 🧠 Model & Training
- **Framework:** MONAI (Medical Open Network for AI)
- **Base Model:** DenseNet121
- **Optimizer:** Adam (LR: 1e-4)
- **Metrics:** Accuracy, ROC AUC, Confusion Matrix

## 🛡️ License
Apache 2.0
