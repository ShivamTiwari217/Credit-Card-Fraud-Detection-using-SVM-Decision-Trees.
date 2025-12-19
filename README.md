# Credit-Card-Fraud-Detection-using-SVM-Decision-Trees.
Credit Card Fraud Detection using Python and Scikit-Learn. Features data preprocessing, class imbalance handling (sample weighting), and model evaluation using ROC-AUC scores.
Context

Credit card fraud remains a significant financial threat, costing banks and consumers billions annually. The core challenge in detecting fraud is the massive "class imbalance"—fraudulent transactions are rare anomalies hidden within millions of legitimate payments. Standard machine learning models often fail in this scenario, as they prioritize overall accuracy by simply predicting "safe" for every transaction, missing the actual fraud.

Goal

The objective of this project is to build a robust machine learning pipeline capable of identifying fraudulent transactions with high precision. By leveraging Scikit-Learn, this project compares two classifiers—Decision Tree and Linear SVC—specifically tuned to handle imbalanced data using sample weighting techniques. The ultimate goal is to maximize the model's ability to catch fraud (High Recall/ROC-AUC) while maintaining a realistic balance of false positives.
