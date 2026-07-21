# Crime Rate Regression

Machine learning models for predicting violent crime rates across U.S. communities using socioeconomic and demographic data.

## Overview

This project explores supervised machine learning techniques for predicting community-level violent crime rates using the UCI Communities and Crime dataset. Multiple regression models and feature engineering methods were implemented and compared to evaluate predictive performance and generalization.

The project investigates how dimensionality reduction, nonlinear feature transformations, and regularization impact model accuracy across different machine learning algorithms.

---

## Dataset

**Source:** UCI Communities and Crime Dataset

- ~2,200 U.S. communities
- ~120 socioeconomic, demographic, and law enforcement features
- Target variable: **ViolentCrimesPerPop**

Features include:

- Household income
- Poverty rate
- Employment
- Education
- Population demographics
- Police staffing
- Housing statistics

---

## Data Preprocessing

The preprocessing pipeline included:

- Removal of identifier variables
- Dropping features with excessive missing values
- Median imputation for missing data
- Standardization using training statistics
- 70/15/15 train-validation-test split
- Log transformation (`log1p`) of the target variable to reduce skewness

---

## Models

Three regression approaches were implemented and evaluated.

### Ridge Regression

Feature spaces:

- Original features
- Polynomial Features (Degree 2)
- PCA (95% explained variance)
- PCA + Degree 3 Polynomial Features

Regularization:

- λ = {0.001, 0.01, 0.1, 1, 10, 100}

---

### Neural Network (PyTorch)

Architecture:

- Fully connected feedforward network
- ReLU activations
- Batch normalization
- Dropout
- Adam optimizer
- Early stopping

Feature spaces:

- Original features
- Polynomial Features
- PCA
- K-Means cluster distance features

---

### K-Nearest Neighbors Regression

Feature spaces:

- Original features
- Polynomial Features
- PCA
- K-Means cluster distance features

Neighborhood sizes:

- k = {1, 3, 5, 10, 20, 50}

---

## Feature Engineering

Several feature engineering techniques were evaluated:

- Principal Component Analysis (PCA)
- Polynomial feature expansion
- K-Means clustering distance embeddings
- Standard scaling
- Log-transformed target variable

The project also examines the effects of multicollinearity, dimensionality reduction, and model complexity on predictive performance.

---

## Results

| Model | Best Validation R² |
|--------|-------------------:|
| Neural Network + PCA | **0.692** |
| Ridge Regression + PCA | **0.689** |
| KNN + PCA | **0.663** |

### Key Findings

- PCA consistently improved generalization across all three models.
- Neural Networks achieved the strongest overall performance.
- Polynomial feature expansion often introduced significant overfitting.
- Appropriate regularization substantially improved model performance in high-dimensional feature spaces.

---

## Technologies

- Python
- PyTorch
- scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Repository Structure

```
.
├── data/
├── notebooks/
├── figures/
├── models/
├── README.md
└── requirements.txt
```

---

## My Contributions

This repository represents a collaborative course project completed as part of NYU's *Introduction to Machine Learning* course.

My contributions included:

- Developing and evaluating regression models
- Implementing preprocessing and feature engineering pipelines
- Performing hyperparameter tuning and model evaluation
- Conducting exploratory data analysis
- Interpreting model performance and experimental results
- Contributing to the written report and project documentation

---

## Team

- Martha McQuillan
- Tyler Huynh

---

## Future Work

Potential improvements include:

- Gradient Boosting (XGBoost / LightGBM)
- Random Forest Regression
- Ensemble learning
- Bayesian hyperparameter optimization
- SHAP feature importance analysis
