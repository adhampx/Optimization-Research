import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
import datetime
import os

# Import custom function from the new module (ensure preprocessing_functions.py is in the same directory)
from preprocessing_functions import apply_log_transform

def drop_high_cardinality(df):
    columns_to_drop = ['srcip', 'dstip'] if set(['srcip', 'dstip']).issubset(df.columns) else []
    return df.drop(columns=columns_to_drop, errors='ignore')

def main():
    # -----------------------------
    # 1. Load Preprocessed Data and Raw CSVs
    # -----------------------------
    pipeline_path = r'D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\\data\\proccessed\\preprocessing_pipeline.joblib'
    train_csv = r'D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\data\\row\\UNSW_NB15_training-set.csv'
    test_csv = r'D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\\data\\row\\UNSW_NB15_testing-set.csv'
    
    # Load the saved preprocessing pipeline
    try:
        preprocessor = joblib.load(pipeline_path)
    except Exception as e:
        print(f"Error loading preprocessing pipeline from {pipeline_path}: {e}")
        return

    # Load raw training and test datasets
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    print("Training dataset preview:")
    print(df_train.head())
    
    target_column = 'label'
    
    # Compute numeric columns from the training dataset (same criteria as used during pipeline creation)
    numeric_cols_list = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_cols_list:
        numeric_cols_list.remove(target_column)
    
    # Set the feature_names attribute for the custom log transformer so it knows which column order to use.
    apply_log_transform.feature_names = numeric_cols_list

    # Separate features and target labels for training and test datasets
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    # Transform the raw features using the preprocessor
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # -----------------------------
    # 2. Hyperparameter Tuning with Bayesian Optimization
    # -----------------------------
    # Define the hyperparameter search space for the Random Forest classifier.
    param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(1, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'criterion': Categorical(['gini', 'entropy'])
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    bayes_search = BayesSearchCV(
        estimator=rf,
        search_spaces=param_space,
        n_iter=30,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    bayes_search.fit(X_train_transformed, y_train)
    best_model = bayes_search.best_estimator_
    print("Best hyperparameters found:", bayes_search.best_params_)
    
    # -----------------------------
    # 3. Model Evaluation
    # -----------------------------
    y_pred = best_model.predict(X_test_transformed)
    y_proba = best_model.predict_proba(X_test_transformed)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    
    original_feature_count = X_train_transformed.shape[1]
    selected_feature_count = X_train_transformed.shape[1]  # no reduction performed
    frr = (original_feature_count - selected_feature_count) / original_feature_count * 100
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Feature Reduction Rate (FRR): {frr:.2f}%")
    
    # -----------------------------
    # 4. Save Performance Measures to CSV
    # -----------------------------
    performance_csv_path = r'D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\\src\\Basic_Models\\Results\\RF.csv'
    
    performance_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "Random Forest",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC_ROC": auc,
        "FPR": fpr,
        "FRR": frr
    }
    perf_df = pd.DataFrame([performance_data])
    
    if os.path.exists(performance_csv_path):
        perf_df.to_csv(performance_csv_path, mode='a', header=False, index=False)
    else:
        perf_df.to_csv(performance_csv_path, index=False)
    print("Performance measures saved in CSV file:", performance_csv_path)

if __name__ == '__main__':
    main()
