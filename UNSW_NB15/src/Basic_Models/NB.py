import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import cross_val_score

def main():
    """
    Main function to train and evaluate a Gaussian Naive Bayes classifier for IDS.
    
    Workflow:
    1. Load preprocessed data
    2. Train Naive Bayes model
    3. Evaluate on test set with various metrics
    4. Save results
    """
    # Define paths
    print("Starting Gaussian Naive Bayes IDS model training...")
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
    
    # Initialize Gaussian Naive Bayes classifier
    # Naive Bayes doesn't have many hyperparameters to tune
    nb = GaussianNB()
    
    # Start training
    print("Training Gaussian Naive Bayes model...")
    start_time = datetime.datetime.now()
    nb.fit(X_train, y_train)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model with cross-validation on the training set
    print("Performing cross-validation...")
    cv_scores = cross_val_score(nb, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Validate on the validation set
    print("Evaluating on validation set...")
    val_pred = nb.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on the test set
    print("Evaluating model on test set...")
    start_time = datetime.datetime.now()
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]  # Probability for the positive class
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
    
    # Create a results dictionary
    results = {
        "Model": "Gaussian Naive Bayes",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    results_path = os.path.join(results_dir, "NB.csv")
    
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
    
    print("\nGaussian Naive Bayes training completed successfully.")

if __name__ == "__main__":
    main() 