# Rice Classification (Cammeo vs Osmancik)

## Project Description
This Machine Learning project aims to classify two varieties of rice (Cammeo and Osmancik) using various morphological features extracted from the grains. The analytical workflow includes data loading, Exploratory Data Analysis (EDA), feature selection, training of various classification models, and interpretation of the results through Explainable AI (XAI) techniques.

## Dataset
The dataset used is `Rice_Cammeo_Osmancik.arff`, which contains several measurements for 3810 rice samples:
* **Numerical features:** `Area`, `Perimeter`, `Major_Axis_Length`, `Minor_Axis_Length`, `Eccentricity`, `Convex_Area`, `Extent`.
* **Target Variable:** `Class` (original labels for Cammeo and Osmancik).

---

## Project Phases

### 1. Data Preprocessing
* Loaded the data in ARFF format using `scipy.io.arff` and converted it into a structured Pandas DataFrame.
* Decoded the strings from byte format to standard `utf-8`.
* Transformed the categorical target variable (`Class`) using One-Hot Encoding, creating two new boolean variables (`Class_Cammeo`, `Class_Osmancik`) and dropping the original column.

### 2. Exploratory Data Analysis (EDA) and Feature Selection
* **Correlation Matrices (Heatmap):** Analyzed linear correlations between continuous features to identify any redundancies using plots generated with Seaborn.
* **Feature Selection for Multicollinearity:** Removed the `Convex_Area` and `Perimeter` variables from the dataset as they are highly correlated with other physical metrics present.
* **Descriptive Statistics:** Generated summary metrics of the cleaned dataset using the `describe()` method.
* **Target Variable Distribution:** Visualized the class count occurrences through a barplot to determine the dataset's balance.

### 3. Model Training and Validation
The project compares three different classification algorithms, evaluating their accuracy using Stratified K-Fold Cross-Validation (K=5 or 10):
* **Random Forest Classifier**.
* **XGBoost Classifier**.
* **Multi-Layer Perceptron (Neural Network):** Implemented within a Scikit-Learn Pipeline that ensures the proper imputation of missing values (`SimpleImputer`) and feature standardization (`StandardScaler`) during the training phase.
* The cross-validation results are extracted and visually compared using a bar chart.

### 4. Explainable AI (XAI) and Advanced Evaluation
* **Confusion Matrix:** Generated to thoroughly examine and evaluate the true/false positives and negatives produced by the Multi-Layer Perceptron model predictions.
* The project imports modules for advanced Explainable AI techniques such as `LIME` (Local Interpretable Model-agnostic Explanations) and `Permutation Importance`, setting up a structured approach for interpreting the models' decisions.
