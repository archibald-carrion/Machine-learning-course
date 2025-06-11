# My Machine Learning and Artificial Intelligence Journey: A Collection of Programs and Experiments

[![Last Commit](https://img.shields.io/github/last-commit/archibald-carrion/Machine-learning-course)](https://github.com/archibald-carrion/Machine-learning-course/commits/main)
[![Open Issues](https://img.shields.io/github/issues/archibald-carrion/Machine-learning-course)](https://github.com/archibald-carrion/Machine-learning-course/issues)

Welcome to my machine learning repository! 
Here you'll find my personal notes and explanations of each used algorithm, along with the code I wrote to implement them, a bunch of resources, and some fun projects I worked on.
This repository is a work in progress, and I plan to add more content over time.

Note: It's important to note that this repository only contains very small projects, to see more complex projects, please check out my other repositories or the section below for related projects.


## 📚 Projects in this Repository

This repository contains a variety of machine learning and AI projects, each in its own folder:

- **basic_neural_network_explanation**: Analysis and visualization of a simple 4-3-2 feedforward neural network, including architecture diagrams and log files.
- **diabetes_decision_tree_optunas**: Decision tree models for diabetes prediction, with hyperparameter tuning using GridSearchCV, Latin Hypercube Sampling, and Optuna. Includes model comparisons and visualizations.
- **diabetes_linear_regression**: Linear regression analysis on diabetes data, with Jupyter notebook explanations and code.
- **diabetes_multiple_regression**: Introduction and cheat sheet for multiple regression, with practical examples in a notebook.
- **mnist_cnn_cluster**: Training and evaluation of a Convolutional Neural Network (CNN) on the MNIST dataset, designed for cluster execution (PyTorch-based).
- **multilayer_perceptron**: Implementation of a Multilayer Perceptron (MLP) from scratch using NumPy, applied to Iris Setosa classification.
- **neural_network**: Implementation of a Convolutional Neural Network (CNN) from scratch using NumPy, focused on MNIST digit classification.
- **RAGs_with_HuggingFace**: Demonstration of Retrieval-Augmented Generation (RAG) using HuggingFace Transformers and FAISS, with a focus on Roman Empire knowledge.
- **titanic_predictive_analysis**: Titanic survival prediction using a variety of machine learning models (Decision Tree, Random Forest, SVM, KNN, Logistic Regression, and more), with data, notebooks, and reports.



## 📂 Repository Structure (Folders)
```plaintext
Machine-learning-course/
├── RAGs_with_HuggingFace
│   ├── README.md
│   └── roman_empire_rag.py
├── README.md
├── basic_neural_network_explanation
│   ├── README.md
│   ├── network.py
│   ├── neural_network_log.txt
│   └── report_neural_network.pdf
├── diabetes_decision_tree_optunas
│   ├── LHS,_Optuna_y_árboles.ipynb
│   ├── README.md
│   ├── arbol_decision_diabetes.png
│   ├── arbol_decision_diabetes_gridsearchcv.png
│   ├── arbol_decision_diabetes_lhs.png
│   ├── arbol_decision_diabetes_optuna.png
│   ├── data
│   ├── diabetes.csv
│   ├── model_comparison_results.csv
│   └── requirements.txt
├── diabetes_linear_regression
│   ├── diabetes.csv
│   └── diabetes_analysis.ipynb
├── diabetes_multiple_regression
│   ├── MultipleRegressionClass.ipynb
│   ├── README.md
│   └── diabetes.csv
├── mnist_cnn_cluster
│   ├── Introducción_a_las_redes_neuronales_convolucionales.ipynb
│   ├── README.md
│   ├── mnist.py
│   ├── mnist_784.csv
│   ├── move_instructions.txt
│   ├── run_mnist.sh
│   └── run_mnist_12479.out
├── multilayer_perceptron
│   ├── Introducción_a_las_redes_neuronales.ipynb
│   └── diabetes.csv
├── neural_network
│   └── Introducción_a_las_redes_neuronales_convolucionales.ipynb
└── titanic_predictive_analysis
    ├── README.md
    ├── data
    ├── models
    ├── notebooks
    ├── reports
    ├── requirements.txt
    └── src
        ├── Titanic_Predictive_Analysis.ipynb
        ├── test.csv
        └── train.csv
```

## Related Projects
- [Convolutional neural network for sound classification](https://github.com/archibald-carrion/Convolutional-neural-networks-sound-classification): Used to detect the sound of an ambulance.
- [Prolog](https://github.com/archibald-carrion/Restaurant-food-composition-system) Expert Systems/Knowledge-Based Systems used to check given on data from a restaurant menu and clients which is the best dish to order.
- [Smart decompiler](https://github.com/archibald-carrion/decompiler) used to decompile code from X86 assembly to a more readble C code.