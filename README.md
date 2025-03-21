# Breast Cancer Prediction

## Overview
This project is a deep learning-based approach to predict breast cancer using the **Breast Cancer Wisconsin Dataset** from **scikit-learn**. The model is built using **PyTorch** and achieves ** 98.25% accuracy** on test data.

## Features
- Uses **Breast Cancer Wisconsin Dataset** for classification.
- Implements a **fully connected neural network (MLP)** using PyTorch.
- **Standardizes** the dataset using `StandardScaler`.
- **Trains and evaluates** the model with accuracy tracking.
- Supports **GPU acceleration** (CUDA-enabled training).
- Displays **test data and model predictions**.

## Technologies Used
- Python
- PyTorch
- scikit-learn
- NumPy

## Dataset
The dataset contains:
- **569 samples** with **30 features** each.
- Two classes: **Malignant (0) & Benign (1)**.
- Available in `sklearn.datasets.load_breast_cancer()`.

## Installation
### Clone the Repository
```bash
 git clone https://github.com/your-username/Breast_Cancer_Prediction.git
 cd Breast_Cancer_Prediction
```

### Install Dependencies
```bash
pip install torch torchvision torchaudio scikit-learn numpy
```

## Usage
### Run the Model
```bash
python breast_cancer_prediction.py
```

## Model Architecture
- **Input Layer**: 30 features
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation

## Training and Evaluation
The model is trained for **100 epochs** using **Binary Cross Entropy Loss (BCELoss)** and **Adam optimizer**.

### Sample Output
```bash
Epoch [100/100], Loss : 0.1084, Accuracy: 97.80%
Accuracy on training data: 97.80%
Accuracy on test data: 98.25%
```

## Test Data & Predictions
The script prints the **test dataset, actual labels, and predicted labels**.




