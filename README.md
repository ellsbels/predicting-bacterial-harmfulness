# Bacteria Identification - MSc Artificial Intelligence and Applications Dissertation 2024 🦠

This repository contains my MSc disseration project, which predicts the harmfulness of bacteria using various characteristics. The main aim of this project was to accurately classify bacteria and assess their potential risk to human health. Supervised machine learning techniques were used to achieve this.

## 📝 Project Overview
- **Title**: Predicting Bacterial Harmfulness: An Evaluation of Machine Learning Models for Classification Based on Family and Environment
- **Research Questions**:
  - Can machine learning algorithms be effectively developed for the task of predicting bacteria as harmful or not harmful based on their family and environment of origin?
  - How do different machine learning algorithms perform in terms of accuracy and reliability when classifying harmful and not harmful bacteria?
  - What is the effect of hyperparameter tuning on model performance, and which model ultimately offers the best balance of accuracy and generalisability?
       
- **Objectives**:
  -	To Develop a Predictive Model: Create and train machine learning models which predict whether bacteria are harmful to humans or not based on their family and location of origin.
  -	To Evaluate Model Performance: Compare the performance of different machine learning algorithms in accurately classifying bacteria as harmful or non-harmful to humans. This study will use accuracy, classification reports, confusion matrixes, and cross-validation scores for evaluation.
  -	To Optimise Model Performance: Enhance the performance of the predictive models through hyperparameter tuning, ultimately selecting the model that achieves the highest accuracy.
    
- **Approach**: The project involved data preprocessing, exploratory data analysis (EDA), feature engineering, and building classification models.

## 📊 Key Insights
- **Techniques Used**: 
  - Exploratory Data Analysis (EDA)
  - Data Cleaning & Feature Engineering
  - Machine Learning Models: Random Forest, Logistic Regression, Support Vector Machine, K-Nearest Neighbours, and Multinomial Naïve Bayes. 
  - Model Evaluation: Accuracy, Precision, Recall, F1-score, and ROC-AUC scores.
  - Model Optimisation: GridSearchCV

## 🛠 Tools & Technologies
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Others**: Jupyter Notebooks for analysis and visualisation

## 📁 Repository Structure
- `notebooks/`: Jupyter notebooks containing the analysis and model-building steps.
- - Full Report is included in the repository as a PDF.

## 📊 Results
- The best-performing model was the K-Nearest Neighbours classifier with an accuracy of 85% and AUC of 0.9211.

## 📜 Future Work
- Expand the dataset with more bacterial samples.
- Experiment with deep learning techniques for better accuracy.

## 🔗 Links
- Dataset: [(https://www.kaggle.com/datasets/kanchana1990/bacteria-dataset)]
