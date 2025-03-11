import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing_functions import apply_log_transform

def drop_high_cardinality(df):
    columns_to_drop = ['srcip', 'dstip'] if set(['srcip', 'dstip']).issubset(df.columns) else []
    return df.drop(columns=columns_to_drop, errors='ignore')

def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    pipeline_path = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\proccessed\preprocessing_pipeline.joblib'
    train_csv = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\row\UNSW_NB15_training-set.csv'
    test_csv = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\row\UNSW_NB15_testing-set.csv'
    
    preprocessor = joblib.load(pipeline_path)
    print("Preprocessing pipeline loaded successfully.")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    print("Training dataset preview:")
    print(df_train.head())
    
    target_column = 'label'
    numeric_cols_list = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_cols_list:
        numeric_cols_list.remove(target_column)
    apply_log_transform.feature_names = numeric_cols_list
    
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        X_train[categorical_cols] = X_train[categorical_cols].astype(str)
        X_test[categorical_cols] = X_test[categorical_cols].astype(str)
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
        print("External categorical encoding applied using OrdinalEncoder.")
    else:
        print("No categorical columns found for external encoding.")
    
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
        X_test_transformed = X_test_transformed.toarray()
    print("X_train_transformed dtype:", X_train_transformed.dtype)
    print("First 5 rows of transformed training data (dense):")
    print(X_train_transformed[:5])
    
    input_dim = X_train_transformed.shape[1]
    model = build_ann_model(input_dim)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train_transformed, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[es], verbose=1)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in skf.split(X_train_transformed, y_train):
        model_cv = build_ann_model(input_dim)
        model_cv.fit(X_train_transformed[train_idx], y_train.iloc[train_idx], epochs=20, batch_size=128, verbose=0)
        score = model_cv.evaluate(X_train_transformed[val_idx], y_train.iloc[val_idx], verbose=0)
        cv_scores.append(score[1])
    print("Cross-validation accuracy scores:", cv_scores)
    print("Mean CV accuracy:", np.mean(cv_scores))
    
    y_pred_prob = model.predict(X_test_transformed)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    original_feature_count = X_train_transformed.shape[1]
    selected_feature_count = X_train_transformed.shape[1]
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
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    sample_idx = np.random.choice(len(y_test), size=10, replace=False)
    if isinstance(y_test, pd.Series):
        actual = y_test.iloc[sample_idx].values
    else:
        actual = y_test[sample_idx]
    print("\nSample Predictions:")
    print("Actual:", actual)
    print("Predicted:", y_pred[sample_idx])
    print("\nEnd-to-End Verification:")
    print("Original training features shape:", X_train.shape)
    print("Transformed training features shape:", X_train_transformed.shape)
    print("Original test features shape:", X_test.shape)
    print("Transformed test features shape:", X_test_transformed.shape)
    performance_csv_path = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\src\Basic_Models\Results\ANN.csv'
    performance_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "ANN",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC-ROC": auc,
        "FPR": fpr,
        "FRR": frr,
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred)
    }
    perf_df = pd.DataFrame([performance_data])
    if os.path.exists(performance_csv_path):
        perf_df.to_csv(performance_csv_path, mode='a', header=False, index=False)
    else:
        perf_df.to_csv(performance_csv_path, index=False)
    print("Performance measures saved in CSV file:", performance_csv_path)

if __name__ == '__main__':
    main()
