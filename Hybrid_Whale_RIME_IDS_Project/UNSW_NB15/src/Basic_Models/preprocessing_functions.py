import pandas as pd
import numpy as np
import hashlib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def log_transform(values):
    """Apply a log1p transformation to handle zero values."""
    return np.log1p(values)

def select_and_log_transform(X, feature_names):
    """
    Apply log transformation on the column 'dbytes', if present.
    
    Parameters:
        X (np.ndarray): The data array (n_samples, n_features).
        feature_names (list): List of feature names corresponding to the columns in X.
        
    Returns:
        np.ndarray: The transformed data array.
    """
    X_transformed = X.copy()
    if 'dbytes' in feature_names:
        idx = feature_names.index('dbytes')
        X_transformed[:, idx] = log_transform(X_transformed[:, idx])
    return X_transformed

def apply_log_transform(X):
    """
    Wrapper function for scikit-learn's FunctionTransformer.
    
    This function applies log transformation on the 'dbytes' column using the feature
    names stored in the function's attribute 'feature_names'. You must set this attribute 
    (e.g., apply_log_transform.feature_names = numeric_cols_list) before building the pipeline.
    
    Parameters:
        X (np.ndarray): The data array to transform.
    
    Returns:
        np.ndarray: The transformed data array.
    
    Raises:
        ValueError: If the attribute 'feature_names' has not been set.
    """
    if not hasattr(apply_log_transform, "feature_names"):
        raise ValueError("The attribute 'feature_names' must be set on apply_log_transform before use.")
    return select_and_log_transform(X, apply_log_transform.feature_names)

def drop_high_cardinality(df):
    """
    Drop high-cardinality columns, such as IP addresses ('srcip' and 'dstip').
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    columns_to_drop = ['srcip', 'dstip'] if set(['srcip', 'dstip']).issubset(df.columns) else []
    return df.drop(columns=columns_to_drop, errors='ignore')

def hash_row(row, exclude_cols=None):
    """
    Compute an MD5 hash of a row's values (converted to a string) after excluding specified columns.
    
    Parameters:
        row (pd.Series): A row from a DataFrame.
        exclude_cols (list): Columns to exclude from the hash.
        
    Returns:
        str: The MD5 hash of the row's values.
    """
    if exclude_cols is None:
        exclude_cols = []
    row_str = ''.join(str(row[col]) for col in row.index if col not in exclude_cols)
    return hashlib.md5(row_str.encode()).hexdigest()

def main():
    # -----------------------------
    # 1. Load Raw Datasets and Verify Data Splitting
    # -----------------------------
    train_csv = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\row\UNSW_NB15_training-set.csv'
    test_csv  = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\row\UNSW_NB15_testing-set.csv'
    
    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)
    
    print("Training dataset preview:")
    print(df_train.head())
    
    # --- Verify Data Splitting and Leakage ---
    if 'id' in df_train.columns and 'id' in df_test.columns:
        id_overlap = df_train['id'].isin(df_test['id']).sum()
        print(f"Overlap count for 'id' between training and test sets: {id_overlap}")
    else:
        print("No 'id' column found; cannot verify overlap using unique IDs.")
    
    exclude_cols = ['id', 'label']
    df_train['row_hash'] = df_train.apply(lambda row: hash_row(row, exclude_cols), axis=1)
    df_test['row_hash']  = df_test.apply(lambda row: hash_row(row, exclude_cols), axis=1)
    
    content_overlap = df_train['row_hash'].isin(df_test['row_hash']).sum()
    print(f"Overlap count based on row content (excluding 'id' and 'label'): {content_overlap}")
    
    target_column = 'label'
    print("Training label distribution:")
    print(df_train[target_column].value_counts())
    print("Testing label distribution:")
    print(df_test[target_column].value_counts())
    
    duplicate_count = df_train.duplicated().sum()
    print(f"Number of duplicate rows in training data: {duplicate_count}")
    
    # -----------------------------
    # 2. Examine Dataset Characteristics and Preprocess
    # -----------------------------
    df_train = drop_high_cardinality(df_train)
    df_test  = drop_high_cardinality(df_test)
    
    # Drop the non-informative 'row_hash' column.
    df_train = df_train.drop(columns=['row_hash'], errors='ignore')
    df_test  = df_test.drop(columns=['row_hash'], errors='ignore')
    
    # Identify numeric columns (excluding the target)
    numeric_cols_list = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_cols_list:
        numeric_cols_list.remove(target_column)
    
    # Identify categorical columns (we force them to string type for encoding).
    categorical_cols_list = df_train.select_dtypes(include=['object']).columns.tolist()
    # Optionally, remove non-informative categorical columns if needed.
    
    print("Numeric columns:", numeric_cols_list)
    print("Categorical columns:", categorical_cols_list)
    
    # Convert categorical columns to string type explicitly (ensures consistency).
    if len(categorical_cols_list) > 0:
        df_train[categorical_cols_list] = df_train[categorical_cols_list].astype(str)
        df_test[categorical_cols_list] = df_test[categorical_cols_list].astype(str)
    
    # Apply label encoding using OrdinalEncoder for categorical columns.
    from sklearn.preprocessing import OrdinalEncoder
    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if len(categorical_cols_list) > 0:
        df_train[categorical_cols_list] = categorical_encoder.fit_transform(df_train[categorical_cols_list])
        df_test[categorical_cols_list] = categorical_encoder.transform(df_test[categorical_cols_list])
    
    # Build a categorical pipeline. Since encoding is already done, only imputation is needed.
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
    ])
    
    # Set the feature names attribute for the custom log transformer.
    apply_log_transform.feature_names = numeric_cols_list
    
    # Create numeric pipeline.
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('log_transform', FunctionTransformer(apply_log_transform, validate=False)),
        ('scaler', MinMaxScaler())
    ])
    
    # Combine pipelines with ColumnTransformer.
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols_list),
        ('cat', categorical_pipeline, categorical_cols_list)
    ])
    
    # Prepare the target label: apply label encoding if necessary.
    if df_train[target_column].dtype == 'object':
        le_target = LabelEncoder()
        df_train[target_column] = le_target.fit_transform(df_train[target_column])
        df_test[target_column] = le_target.transform(df_test[target_column])
        print("Label encoding applied on target column in training dataset.")
    else:
        print("Target column is already numeric in training dataset.")
    
    # Separate features and target labels.
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test  = df_test.drop(columns=[target_column])
    y_test  = df_test[target_column]
    
    # -----------------------------
    # 3. Fit the Preprocessing Pipeline and Inspect Transformations
    # -----------------------------
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed  = preprocessor.transform(X_test)
    
    print("Summary statistics for transformed training data:")
    try:
        _ = X_train_transformed.tocsc()  # Check if sparse.
        sparse_df = pd.DataFrame.sparse.from_spmatrix(X_train_transformed)
    except AttributeError:
        sparse_df = pd.DataFrame(X_train_transformed)
    
    sample_df = sparse_df.sample(n=1000, random_state=42)
    print(sample_df.describe())
    
    # -----------------------------
    # 4. Save the Preprocessing Pipeline
    # -----------------------------
    pipeline_save_path = r'D:\Optimization-Research\Hybrid_Whale_RIME_IDS_Project\UNSW_NB15\data\proccessed\preprocessing_pipeline.joblib'
    joblib.dump(preprocessor, pipeline_save_path)
    print("Preprocessing pipeline saved as 'preprocessing_pipeline.joblib'.")

if __name__ == '__main__':
    main()
