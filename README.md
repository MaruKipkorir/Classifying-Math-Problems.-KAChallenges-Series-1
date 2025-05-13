# Classifying-Math-Problems.-KAChallenges-Series-1

# TF-IDF Based Text Classification with an Ensemble of LightGBM, SVM, and Naive Bayes.

This project aims to classify math problems written in natural language into one of eight predefined categories, such as algebra, calculus, and linear algebra. The task is part of a machine learning competition where the goal is to predict the correct topic for each given math problem. [KAChallenges Series 1: Classifying Math Problems](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/overview)

## Problem Overview

Given a math problem written in natural language, we use machine learning techniques to predict the appropriate category from a set of eight predefined categories.

## Dataset
[View on Kaggle](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/data)
The dataset consists of math problems written in natural language. It contains two CSV files:

* **train.csv**: Contains the labeled training data.
* **test.csv**: Contains the test data on which predictions are made.

Each entry in the dataset contains:

* **Question**: The math problem written in natural language.
* **Label**: The correct category of the problem.

## Approach
Text Representation: Questions were transformed into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This method captures the importance of words (and n-grams up to trigrams) within the dataset while down-weighting common but less informative terms.

Vectorization Details:

ngram_range=(1, 3) (unigrams, bigrams, trigrams)

max_features=100,000 (top tokens by frequency)

## Steps Involved

### 1. Data Preprocessing

The following steps are applied to preprocess the text data:

* **LaTeX Conversion**: The questions are in LaTeX format, which is converted into plain text using `pylatexenc`.
* **Text Cleaning**: The text is cleaned by removing periods, tokenizing, converting to lowercase, removing stopwords, and lemmatizing the tokens.
* **Feature Extraction**: The cleaned text is transformed into TF-IDF feature vectors using unigrams, bigrams, and trigrams.

### 2. Hyperparameter Optimization with Optuna

We perform hyperparameter tuning using **Optuna**, a hyperparameter optimization library. The objective is to find the best combination of hyperparameters for different models, such as:

* **LightGBM (LGBM)**: A gradient boosting framework.
* **Multinomial Naive Bayes (MNB)**: A probabilistic classifier for text data.
* **Linear Support Vector Classifier (SVM)**: A classification model based on SVM.

Optuna searches for the best hyperparameters using **Stratified K-Fold cross-validation** and aims to maximize the **micro-averaged F1 score**.

### 3. Model Training and Evaluation

We train the best-performing models from Optuna and evaluate them on the training set using **F1 Score**. The models are then combined into an ensemble model using **VotingClassifier**, which includes:

* **LGBM**
* **SVM**
* **Naive Bayes**

### 4. Prediction and Submission

The trained ensemble model is used to predict the labels for the test dataset. The results are saved into a CSV file for submission.

### 6. Submission

The final submission consists of the **ID** and predicted **label** for each test sample, saved into a CSV file, ready for evaluation.


### Conclusion
This model has an F1 score of 0.7960. It makes for a decent baseline. More advanced approaches such as deep learning architectures, could provide better performance.
