# Weather Classification System

A machine learning system that classifies outdoor images into 5 weather conditions — Sunny, Rainy, Cloudy, Foggy, and Snowy — using classical computer vision feature extraction and ensemble learning.

## Overview
Instead of relying on deep learning, this project demonstrates that carefully engineered image features combined with strong classical ML models can achieve solid classification performance on a real-world visual task.

## Approach
- *Data Augmentation*: horizontal flip, brightness variation, and slight rotation to expand the training set
- *Feature Extraction*: 
  - HOG (Histogram of Oriented Gradients) — edge/gradient patterns
  - Local Binary Patterns (LBP) — texture information
  - HSV color histograms — sky color and brightness distribution
  - Gabor filters — frequency and orientation-based texture
- *Dimensionality Reduction*: PCA (reduced from 8,246 to 294 features)
- *Model*: SVM tuned via GridSearchCV, combined into a Voting Ensemble with Random Forest and Gradient Boosting

## Results
- Best CV Accuracy: *85.21%*
- Test Accuracy: *78.33%*
- Evaluated with a full classification report and confusion matrix

## Tech Stack
Python, OpenCV, scikit-image, scikit-learn, NumPy, Matplotlib, Seaborn

## Files
- Weather_Classification_System.ipynb — complete notebook (data loading, feature extraction, training, evaluation, and prediction demo)
