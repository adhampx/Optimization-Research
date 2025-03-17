import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, cross_val_score

def main():
    """
    Main function to train and evaluate a K-Nearest Neighbors classifier for IDS.
    
    Workflow:
    1. Load preprocessed data
    2. Train KNN with hyperparameter tuning
    3. Evaluate on test set with various metrics
    4. Save results
    """
    # Define paths
    print("Starting K-Nearest Neighbors IDS model training...")
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
    
    # Given the large dataset size, we'll use a subset for training to make KNN more efficient
    # This is often necessary for KNN which can be computationally expensive on large datasets
    sample_size = min(100000, X_train.shape[0])  # Use at most 100,000 samples
    random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_train_sampled = X_train.iloc[random_indices]
    y_train_sampled = y_train.iloc[random_indices]
    
    print(f"Using {sample_size} samples for KNN training")
    
    # Set up hyperparameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
    }
    
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_jobs=-1)
    
    # Set up GridSearchCV
    print("Starting hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,  # Reduced from 5 to 3 due to computational intensity
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model on the sampled training data
    grid_search.fit(X_train_sampled, y_train_sampled)
    
    # Get the best model
    best_knn = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")
    
    # Evaluate model with cross-validation on the sampled training set
    cv_scores = cross_val_score(best_knn, X_train_sampled, y_train_sampled, cv=3, scoring='accuracy')
    print(f"Cross-validation accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Validate on the validation set
    val_pred = best_knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on the test set
    print("Evaluating model on test set...")
    y_pred = best_knn.predict(X_test)
    y_prob = best_knn.predict_proba(X_test)[:, 1]  # Probability for the positive class
    
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
    
    # Create a results dictionary
    results = {
        "Model": "K-Nearest Neighbors",
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
        "Sample_Size": sample_size
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(results_dir, "KNN.csv")
    
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
    
    print("\nK-Nearest Neighbors training completed successfully.")

if __name__ == "__main__":
    main() 