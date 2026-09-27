# Module 2 — Analytics (Titanic Dataset)

## Overview
Titanic dataset meeda EDA, statistical analysis, classification, regression, imbalance handling chesamu.

## Key Steps
- EDA + missing value handling (age: median impute, embarked: drop, deck: drop — 77% missing)
- Univariate/bivariate/multivariate analysis with interpretation
- Stratified 80/20 train-test split
- 3 classifiers: Logistic Regression, Decision Tree, Random Forest — confusion matrix, ROC-AUC, comparison table
- Imbalance handling: baseline vs class_weight vs SMOTE (SMOTE best)
- GridSearchCV tuning on Random Forest (OOB=0.809)
- Linear regression for fare prediction (R²=0.325)
- Final pipeline saved with joblib

## Recommendation
SMOTE + tuned Random Forest recommended for deployment.

## Files
- notebooks/zepto_analytics.ipynb
- data/titanic.csv
- titanic_pipeline.pkl
