# Stroke Risk Identification Using Supervised Machine Learning Models

A Python-based machine learning project that identifies stroke risk using multiple supervised machine learning algorithms.

## Project Overview

This project uses healthcare-related data to classify records as **Stroke** or **No Stroke**. The system preprocesses the dataset, trains multiple machine learning models, evaluates their performance, and predicts outcomes for test data.

## Machine Learning Algorithms

- Naive Bayes
- J48 / Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- Artificial Neural Network (ANN)

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Tkinter

## Key Features

- Upload CSV dataset
- Data preprocessing and missing-value handling
- Categorical data encoding
- 80:20 train-test split
- Train multiple machine learning models
- Accuracy, Precision, Recall and F1 Score evaluation
- Confusion Matrix visualization
- Algorithm performance comparison
- Stroke / No Stroke prediction
- Tkinter graphical user interface

## Project Workflow

Dataset → Preprocessing → Encoding → Train/Test Split → Model Training → Evaluation → Model Comparison → Prediction

## Evaluation Metrics

The implemented models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Keep `dataset.csv` and `testData.csv` in the project folder.
4. Run:

```bash
python StrokeDetectionnn.py
