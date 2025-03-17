import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, cross_val_score

def main():
    """
    Main function to train and evaluate a Support Vector Machine classifier for IDS.
    
    Workflow:
    1. Load preprocessed data
    2. Train SVM with hyperparameter tuning
    3. Evaluate on test set with various metrics
    4. Save results
    """
    # Define paths
    print("Starting Support Vector Machine IDS model training...")
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
    
    # Given the large dataset size, we'll use a small subset for training SVM
    # SVMs do not scale well to large datasets
    sample_size = min(10000, X_train.shape[0])  # Use at most 10,000 samples for SVM
    random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_train_sampled = X_train.iloc[random_indices]
    y_train_sampled = y_train.iloc[random_indices]
    
    print(f"Using {sample_size} samples for SVM training due to computational constraints")
    
    # Set up hyperparameter grid for SVM
    # Using a simplified grid due to computational complexity
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Initialize SVM classifier
    svm = SVC(probability=True, random_state=42)
    
    # Set up GridSearchCV
    print("Starting hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,  # Reduced CV to minimize computational load
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model on the sampled training data
    grid_search.fit(X_train_sampled, y_train_sampled)
    
    # Get the best model
    best_svm = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")
    
    # Evaluate model with cross-validation on the sampled training set
    cv_scores = cross_val_score(best_svm, X_train_sampled, y_train_sampled, cv=3, scoring='accuracy')
    print(f"Cross-validation accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Use a subset of validation data to speed up evaluation
    val_sample_size = min(20000, X_val.shape[0])
    val_indices = np.random.choice(X_val.shape[0], val_sample_size, replace=False)
    X_val_sampled = X_val.iloc[val_indices]
    y_val_sampled = y_val.iloc[val_indices]
    
    # Validate on the sampled validation set
    val_pred = best_svm.predict(X_val_sampled)
    val_accuracy = accuracy_score(y_val_sampled, val_pred)
    print(f"Validation accuracy (on {val_sample_size} samples): {val_accuracy:.4f}")
    
    # Use a subset of test data to speed up evaluation
    test_sample_size = min(20000, X_test.shape[0])
    test_indices = np.random.choice(X_test.shape[0], test_sample_size, replace=False)
    X_test_sampled = X_test.iloc[test_indices]
    y_test_sampled = y_test.iloc[test_indices]
    
    # Evaluate on the sampled test set
    print(f"Evaluating model on {test_sample_size} test samples...")
    y_pred = best_svm.predict(X_test_sampled)
    y_prob = best_svm.predict_proba(X_test_sampled)[:, 1]  # Probability for the positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_sampled, y_pred)
    precision = precision_score(y_test_sampled, y_pred, zero_division=0)
    recall = recall_score(y_test_sampled, y_pred, zero_division=0)
    f1 = f1_score(y_test_sampled, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test_sampled, y_prob)
    
    # Calculate confusion matrix and derived metrics
    tn, fp, fn, tp = confusion_matrix(y_test_sampled, y_pred).ravel()
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
    
    # Create a results dictionary
    results = {
        "Model": "Support Vector Machine",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Best_Parameters": str(grid_search.best_params_),
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
        "Training_Sample_Size": sample_size,
        "Validation_Sample_Size": val_sample_size,
        "Test_Sample_Size": test_sample_size
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(results_dir, "SVM.csv")
    
    # Check if the results file already exists
    if os.path.exists(results_path):
        # Append without writing the header
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        # Create a new file with header
        results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    
    # Sample manual inspection
    print("\nSample Predictions (First 10 test instances from sampled test set):")
    sample_indices = np.arange(min(10, y_test_sampled.shape[0]))
    sample_actual = y_test_sampled.iloc[sample_indices].values
    sample_pred = y_pred[sample_indices]
    
    for i, (actual, pred) in enumerate(zip(sample_actual, sample_pred)):
        print(f"Sample {i+1}: Actual={actual}, Predicted={pred}, {'Correct' if actual == pred else 'Incorrect'}")
    
    print("\nSupport Vector Machine training completed successfully.")

if __name__ == "__main__":
    main() 