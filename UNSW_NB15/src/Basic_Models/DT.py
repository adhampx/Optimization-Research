import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
import datetime
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder  # For external categorical encoding

# Import custom function from the new module (ensure preprocessing_functions.py is in the same directory)
from preprocessing_functions import apply_log_transform

def drop_high_cardinality(df):
    columns_to_drop = ['srcip', 'dstip'] if set(['srcip', 'dstip']).issubset(df.columns) else []
    return df.drop(columns=columns_to_drop, errors='ignore')

def main():
    # -----------------------------
    # 1. Load Preprocessed Data and Raw CSVs
    # -----------------------------
    pipeline_path = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\proccessed\preprocessing_pipeline.joblib'
    train_csv = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\row\UNSW_NB15_training-set.csv'
    test_csv = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\row\UNSW_NB15_testing-set.csv'
    
    try:
        preprocessor = joblib.load(pipeline_path)
        print("Preprocessing pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading preprocessing pipeline from {pipeline_path}: {e}")
        return

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    print("Training dataset preview:")
    print(df_train.head())
    
    target_column = 'label'
    
    # Compute numeric columns (as used when building the pipeline)
    numeric_cols_list = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_cols_list:
        numeric_cols_list.remove(target_column)
    
    # Set the feature names attribute for the custom log transformer
    apply_log_transform.feature_names = numeric_cols_list
    
    # Separate features and target labels
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    # ----- NEW STEP: External Categorical Encoding -----
    # Identify categorical columns (as read from CSV)
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        # Convert to string explicitly
        X_train[categorical_cols] = X_train[categorical_cols].astype(str)
        X_test[categorical_cols] = X_test[categorical_cols].astype(str)
        # Apply OrdinalEncoder to convert categorical values to numeric.
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
        print("External categorical encoding applied using OrdinalEncoder.")
    else:
        print("No categorical columns found for external encoding.")
    # ----------------------------------------------------
    
    # Transform raw features using the loaded pipeline
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Debug: Verify that transformed training data is numeric.
    print("X_train_transformed dtype:", X_train_transformed.dtype)
    print("First 5 rows of transformed training data (dense):")
    if hasattr(X_train_transformed, "toarray"):
        print(X_train_transformed.toarray()[:5])
    else:
        print(X_train_transformed[:5])
    
    # -----------------------------
    # 2. Hyperparameter Tuning with Bayesian Optimization
    # -----------------------------
    param_space = {
        'max_depth': Integer(1, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'criterion': Categorical(['gini', 'entropy'])
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    
    bayes_search = BayesSearchCV(
        estimator=dt,
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
    # 3. Evaluate Model Overfitting using Cross-Validation
    # -----------------------------
    cv_scores = cross_val_score(best_model, X_train_transformed, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print("Cross-validation accuracy scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())
    
    # -----------------------------
    # 4. Model Evaluation: Predictions and Metrics
    # -----------------------------
    y_pred = best_model.predict(X_test_transformed)
    y_proba = best_model.predict_proba(X_test_transformed)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
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
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # -----------------------------
    # 5. Manual Inspection of Predictions
    # -----------------------------
    sample_idx = np.random.choice(len(y_test), size=10, replace=False)
    if isinstance(y_test, pd.Series):
        actual = y_test.iloc[sample_idx].values
    else:
        actual = y_test[sample_idx]
    print("\nSample Predictions:")
    print("Actual:", actual)
    print("Predicted:", y_pred[sample_idx])
    
    # -----------------------------
    # 6. End-to-End Verification
    # -----------------------------
    print("\nEnd-to-End Verification:")
    print("Original training features shape:", X_train.shape)
    print("Transformed training features shape:", X_train_transformed.shape)
    print("Original test features shape:", X_test.shape)
    print("Transformed test features shape:", X_test_transformed.shape)
    
    # -----------------------------
    # 7. Save Performance Measures to CSV
    # -----------------------------
    performance_csv_path = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\src\Basic_Models\Results\DT.csv'
    performance_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "Decision Tree",
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
