## Project Overview
This project aims to develop a machine learning-based fraud detection model to identify fraudulent transactions in a financial transaction dataset.

## Dataset
The project utilizes a publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/goyaladi/fraud-detection-dataset/data) that contains financial transactions and their corresponding labels (whether the transaction is fraudulent or not). The dataset consists of 1,000 transactions, with 45 fraudulent transactions (4.5% of the total transactions).

## Methodology
The project follows a standard machine learning workflow, including explanatory data analysis, data preprocessing and feature engineering, model training and selection, and model evaluation. The following techniques are used:

1. **Explanatory data analysis**: utilize visualization technique to reveal preliminary pattern and understand the potential important features.
2. **Feature Engineering**: Extracting relevant features from (i) transaction timestamp, such as office hour indicator, weekday, Hour etc; (ii) Transaction Amount and Account Balance to create transaction amount to account balance ratio.
3. **Model Training and Selection**: Use Synthetic Minority Oversampling Technique (SMOTE) to oversample the minority class, experimenting with various machine learning algorithms, including Logistic Regression, Random Forest, and XGBoost, to identify the best-performing model.
4. **Model Evaluation**: Assessing the model's performance using recall, and interpret the results to identify the most important factors.
