import pandas as pd
import numpy as np
import joblib
import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)

def build_cnn_model(input_shape):
    """
    Build a simple CNN model for binary classification.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features)
        
    Returns:
        A compiled CNN model
    """
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """
    Main function to train and evaluate a CNN model for IDS.
    
    Workflow:
    1. Load preprocessed data
    2. Reshape the data for CNN input (add channel dimension)
    3. Train CNN with fixed parameters
    4. Evaluate on test set with various metrics
    5. Save results
    """
    print("Starting CNN IDS model training...")
    
    # Define paths
    # Use the new preprocessed data paths
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
    
    # For CNN, we need to reshape the data to (samples, timesteps, features)
    # For 1D CNN, we'll treat each feature as a timestep with 1 channel
    # X_train_cnn shape: (n_samples, n_features, 1)
    X_train_cnn = np.expand_dims(X_train.values, axis=2)
    X_val_cnn = np.expand_dims(X_val.values, axis=2)
    X_test_cnn = np.expand_dims(X_test.values, axis=2)
    
    print(f"CNN input shape: {X_train_cnn.shape}")
    
    # Define the input shape for the CNN
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
    
    # Build and compile the CNN model
    print("Building CNN model...")
    model = build_cnn_model(input_shape)
    model.summary()
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    print("Training the CNN model...")
    start_time = datetime.datetime.now()
    
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=30,
        batch_size=256,
        callbacks=[early_stopping],
        verbose=1
    )
    
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    start_time = datetime.datetime.now()
    
    # Make predictions
    y_prob = model.predict(X_test_cnn)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    
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
        "Model": "CNN",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    results_path = os.path.join(results_dir, "CNN.csv")
    
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
    
    if isinstance(y_test, pd.Series):
        sample_actual = y_test.iloc[sample_indices].values
    else:
        sample_actual = y_test[sample_indices]
        
    sample_pred = y_pred[sample_indices]
    
    for i, (actual, pred) in enumerate(zip(sample_actual, sample_pred)):
        print(f"Sample {i+1}: Actual={actual}, Predicted={pred}, {'Correct' if actual == pred else 'Incorrect'}")
    
    print("\nCNN training completed successfully.")

if __name__ == "__main__":
    main()
