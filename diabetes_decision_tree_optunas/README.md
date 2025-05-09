# Decision Trees and Hyperparameter Tuning for Diabetes Prediction

This project demonstrates the implementation and comparison of various hyperparameter tuning techniques for decision tree models on the Indian Diabetes dataset from Kaggle. The project focuses on comparing linear regression approaches with decision tree models tuned using different optimization strategies.

## Project Overview

```Mermaid
flowchart TD
    A[Data Loading & Preprocessing] --> B[Split Data 85%-15%]
    B --> C[Linear Regression Models]
    B --> D[Decision Tree Models]
    
    C --> C1[LASSO]
    C --> C2[Ridge]
    C --> C3[Elastic Net]
    
    D --> D1[Default Decision Tree]
    D --> D2[Hyperparameter Tuning]
    
    D2 --> D2a[Latin Hypercube Sampling\n90 samples]
    D2 --> D2b[Optuna\n90 trials]
    D2 --> D2c[GridSearchCV]
    
    D2a --> E[Model Evaluation]
    D2b --> E
    D2c --> E
    C1 --> E
    C2 --> E
    C3 --> E
    D1 --> E
    
    E --> E1[Accuracy]
    E --> E2[Confusion Matrix]
    E --> E3[Classification Report]
    E --> E4[Tree Visualization]
    
    E1 --> F[Comparison & Analysis]
    E2 --> F
    E3 --> F
    E4 --> F
```

In this project, we:

1. Load and preprocess the Indian Diabetes dataset
2. Implement linear regression models (LASSO, Ridge, Elastic Net)
3. Create a baseline decision tree model
4. Tune hyperparameters using three different methods:
   - Latin Hypercube Sampling (LHS)
   - Optuna optimization framework
   - GridSearchCV from Scikit-learn
5. Compare and evaluate all models using classification metrics

## Requirements

To run this project, you need Python 3.6+ with the following packages:

```
pandas
numpy
scikit-learn
matplotlib
optuna
```

## Installation & Usage

Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Indian Diabetes dataset from Kaggle, which contains various health metrics and a binary target variable indicating whether the patient has diabetes.

Features include:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Hyperparameter Tuning Approaches

### 1. Latin Hypercube Sampling (LHS)

### 2. Optuna

### 3. GridSearchCV

## Results and Comparison
