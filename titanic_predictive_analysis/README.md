


# Titanic Predictive Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-blue.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-blue.svg)](https://seaborn.pydata.org/)

## 📋 Overview

This project uses machine learning algorithms from scikit-learn to analyze survival patterns of Titanic passengers. By examining various passenger attributes such as age, gender, ticket class, and family relationships, we build predictive models to understand the factors that influenced survival rates during this historic disaster.

<!-- ## 🧪 Project Workflow -->




## 🔍 Features

- **Data Preprocessing**: Handle missing values, encode categorical variables, and prepare data for machine learning
- **Exploratory Data Analysis**: Visualize relationships between passenger features and survival rates
- **Feature Engineering**: Create new features that might better predict survival (e.g., Title)
- **Machine Learning Models**: Implement and compare multiple classification algorithms:
    - Decision Tree
    - SGD Classifier Training
    - Random Forest Training
    - Logistic Regression Training
    - KNN Training
    - Gaussian Naive Bayes Training
    - Perceptron Training
    - SVM Training

## 🚀 Installation & Setup

1. Clone this repository:
```bash
git clone https://github.com/archibald-carrion/Titanic-predictive-analysis.git
cd Titanic-predictive-analysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter notebook:
You can run the Jupyter notebooks in your environment of choice, but it was developed using VS Code insiders and the Jupyter extension.

## 📂 Project Structure

The project directory is organized as follows:

```
Titanic-predictive-analysis/
│
├── data/                           # Data directory
│   ├── train.csv                   # Training dataset
│   └── test.csv                    # Test dataset
│
├── notebooks/                      # Jupyter notebooks
│   ├── Titanic_Predictive_Analysis.ipynb  # Main analysis notebook
│   └── EDA.ipynb                   # Exploratory data analysis
│
├── src/                            # Source code
│   ├── data_processing.py          # Data preprocessing utilities
│   ├── feature_engineering.py      # Feature creation and selection
│   └── model_evaluation.py         # Model evaluation utilities
│
├── models/                         # Saved model files
│
├── reports/                        # Generated analysis as HTML, PDF, etc.
│   └── figures/                    # Generated graphics and figures
│
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## TODO
- [ ] Refactor code for better readability
- [ ] Use multiple notebooks for different analyses
- [ ] Refactor the project structure for better organization
- [ ] Add content to requirements.txt

## 🙏 Acknowledgments

<!-- - [Kaggle](https://www.kaggle.com/c/titanic) for providing the Titanic dataset -->
- The scikit-learn team for their excellent machine learning library
- The Jupyter project for making data science more accessible
