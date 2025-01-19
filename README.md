# Credit Card Fraud Detection 🛡️

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-latest-orange)

## Project Overview 🎯

This project implements a credit card fraud detection system using machine learning techniques. The system analyzes transaction data to identify potentially fraudulent activities, focusing on handling imbalanced data and optimizing for recall and precision in fraud detection.

## Authors 👥

- Xiang Liu
- Mabel Mires 
- Natalia Benitez

## Project Structure 📁

```
.
├── TrabajoFinal.ipynb                                        # Main notebook with model implementations and analysis
├── MemoriaFinal_XiangLiu_MabelMires_NataliaBenitez.pdf      # Final project report with detailed methodology
├── Comparar modelos.xlsx                                     # Comparative analysis of model performances
├── README.md                                                 # Project documentation and setup guide
└── requirements.txt                                          # Python package dependencies

Note: The dataset (creditcard.csv) needs to be downloaded from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

## Key Features ⭐

- Data preprocessing using StandardScaler and RobustScaler
- Principal Component Analysis (PCA) for feature selection
- Handling imbalanced data using:
  - Random Undersampling
  - Random Oversampling
  - SMOTE (Synthetic Minority Over-sampling Technique)
- Cross-validation with StratifiedKFold
- Hyperparameter optimization using GridSearchCV

## Models Implementation 🤖

The project implements six different machine learning models for fraud detection:

1. Logistic Regression
   - Basic implementation with default parameters
   - Tested with balanced and imbalanced datasets

2. Artificial Neural Network (ANN)
   - Using TensorFlow/Keras
   - Binary cross-entropy loss
   - Adam optimizer

3. K-Nearest Neighbors (KNN)
   - Tested with different values of k
   - Optimized for imbalanced data
   - Implemented with both under and over-sampling

4. Support Vector Machine (SVM)
   - Using both linear and non-linear kernels
   - Optimized with GridSearchCV
   - Tested with different sampling techniques

5. XGBoost
   - Optimized for imbalanced classification
   - High scalability and accuracy
   - Tested with under and over-sampling approaches

6. Random Forest
   - Ensemble learning method
   - Tested with balanced and imbalanced datasets
   - Optimized with different sampling techniques

Each model was evaluated using three different approaches:
- Original imbalanced dataset
- Random Under-Sampling
- Random Over-Sampling

## Model Evaluation 📊

Models are evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Confusion Matrix

Special emphasis is placed on recall and precision for fraud detection (class 1), as these are critical metrics for fraud detection systems.

## Installation 💻

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- tensorflow
- xgboost
- imbalanced-learn
- matplotlib
- seaborn

## Usage 🚀

1. Open `TrabajoFinal.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load and preprocess data
   - Train models with different sampling techniques
   - Evaluate and compare model performance

## Results 📈

Detailed performance metrics and model comparisons can be found in `Comparar modelos.xlsx`. The evaluation focuses on:
- Model performance on imbalanced data
- Impact of different sampling techniques
- Trade-off between precision and recall
- Overall detection effectiveness

## License ⚖️

MIT License

Copyright (c) 2024 Xiang Liu, Mabel Mires, Natalia Benitez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
