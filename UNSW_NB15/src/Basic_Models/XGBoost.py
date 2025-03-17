import pandas as pd
import numpy as np
import joblib
import os
import datetime
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import cross_val_score

def main():
    """
    Main function to train and evaluate an XGBoost classifier for IDS.
    
    Workflow:
    1. Load preprocessed data
    2. Train XGBoost with predefined parameters
    3. Evaluate on test set with various metrics
    4. Save results
    """
    # Define paths
    print("Starting XGBoost IDS model training...")
    preprocessor_path = "D:/Optimization-Research/UNSW_NB15/data/processed/preprocessing_pipeline.joblib"
    train_data_path = "D:/Optimization-Research/UNSW_NB15/data/processed/Training_dataset.csv"
    val_data_path = "D:/Optimization-Research/UNSW_NB15/data/processed/Validation_dataset.csv"
    test_data_path = "D:/Optimization-Research/UNSW_NB15/data/processed/Testing_dataset.csv"
    results_dir = "D:/Optimization-Research/UNSW_NB15/src/Basic_Models/Results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("Loading preprocessed data...")
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    
    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label']
    
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize XGBoost classifier with predefined parameters
    # These parameters are chosen based on general best practices for binary classification
    xgb_params = {
        'n_estimators': 100,       # Number of boosting rounds
        'max_depth': 5,            # Maximum tree depth
        'learning_rate': 0.1,      # Learning rate
        'subsample': 0.8,          # Subsample ratio of the training instances
        'colsample_bytree': 0.8,   # Subsample ratio of columns when constructing each tree
        'min_child_weight': 1,     # Minimum sum of instance weight needed in a child
        'gamma': 0,                # Minimum loss reduction required to make a further partition
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'auc',      # Evaluation metric
        'random_state': 42,        # Random seed
        'n_jobs': -1               # Use all CPU cores
    }
    
    # Create XGBoost classifier
    print("Training XGBoost model with predefined parameters...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Start training
    start_time = datetime.datetime.now()
    
    # Simple training without early stopping
    xgb_model.fit(X_train, y_train)
    
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model with cross-validation on the training set
    print("Performing cross-validation...")
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='accuracy')
    print(f"Cross-validation accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Validate on the validation set
    print("Evaluating on validation set...")
    val_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on the test set
    print("Evaluating model on test set...")
    start_time = datetime.datetime.now()
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]  # Probability for the positive class
    end_time = datetime.datetime.now()
    testing_time = (end_time - start_time).total_seconds()
    print(f"Testing completed in {testing_time:.2f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    
    # Calculate confusion matrix and derived metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)  # False Positive Rate
    frr = fn / (fn + tp)  # False Rejection Rate (or False Negative Rate)
    
    # Print evaluation metrics
    print("\nTest Set Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # Feature importance analysis
    feature_importance = xgb_model.feature_importances_
    feature_names = X_train.columns
    
    # Sort feature importances
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Create a results dictionary
    results = {
        "Model": "XGBoost",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Parameters": str(xgb_params),
        "Cross_Val_Accuracy_Mean": cv_scores.mean(),
        "Cross_Val_Accuracy_Std": cv_scores.std(),
        "Validation_Accuracy": val_accuracy,
        "Test_Accuracy": accuracy,
        "Test_Precision": precision,
        "Test_Recall": recall,
        "Test_F1_Score": f1,
        "Test_AUC_ROC": auc_roc,
        "Test_FPR": fpr,
        "Test_FRR": frr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Training_Time": training_time,
        "Testing_Time": testing_time
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(results_dir, "XGBoost.csv")
    
    # Check if the results file already exists
    if os.path.exists(results_path):
        # Append without writing the header
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        # Create a new file with header
        results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    
    # Sample manual inspection
    print("\nSample Predictions (First 10 test instances):")
    sample_indices = np.arange(10)
    sample_actual = y_test.iloc[sample_indices].values
    sample_pred = y_pred[sample_indices]
    
    for i, (actual, pred) in enumerate(zip(sample_actual, sample_pred)):
        print(f"Sample {i+1}: Actual={actual}, Predicted={pred}, {'Correct' if actual == pred else 'Incorrect'}")
    
    print("\nXGBoost training completed successfully.")

if __name__ == "__main__":
    main()
