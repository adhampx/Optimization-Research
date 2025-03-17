import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def log_transform(X):
    """Apply log(x+1) transformation to handle zero values."""
    return np.log1p(X)

def select_and_log_transform(X, feature_names, skewed_features):
    """
    Apply log transformation to skewed numeric features.
    
    Parameters:
        X (np.ndarray): The data array (n_samples, n_features).
        feature_names (list): List of feature names corresponding to the columns in X.
        skewed_features (list): List of feature names that are skewed and need log transformation.
        
    Returns:
        np.ndarray: The transformed data array.
    """
    X_transformed = X.copy()
    for feature in skewed_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            X_transformed[:, idx] = log_transform(X_transformed[:, idx])
    return X_transformed

# Define a global transformation function to work with pickling
def transform_func(X):
    # This will be set later with the correct parameters
    return X

class LogTransformer(FunctionTransformer):
    """
    Custom transformer to apply log transformation to specified features.
    """
    def __init__(self, feature_names=None, skewed_features=None):
        self.feature_names = feature_names if feature_names is not None else []
        self.skewed_features = skewed_features if skewed_features is not None else []
        super().__init__(func=transform_func, validate=False)
    
    def fit(self, X, y=None):
        global transform_func
        # Update the global transformation function
        transform_func = lambda X_inner: select_and_log_transform(
            X_inner, self.feature_names, self.skewed_features
        )
        return super().fit(X, y)
    
    def set_feature_names(self, feature_names):
        self.feature_names = feature_names
        return self

def drop_high_cardinality(df, columns_to_drop=None):
    """
    Drop high-cardinality or non-informative columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns_to_drop (list): List of column names to drop.
        
    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    if columns_to_drop is None:
        columns_to_drop = ['srcip', 'dstip', 'Stime', 'Ltime']
    
    # Check which columns actually exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    return df.drop(columns=columns_to_drop, errors='ignore')

def preprocess_data():
    """
    Main function to preprocess the UNSW-NB15 dataset.
    
    Steps:
    1. Load raw data
    2. Assign column names
    3. Drop high-cardinality columns
    4. Handle missing values
    5. Encode categorical features
    6. Apply log transformation to skewed numeric features
    7. Scale numeric features
    8. Split data into training, validation, and testing sets
    9. Save the preprocessing pipeline and the processed datasets
    
    Returns:
        None
    """
    # Define paths
    raw_data_path = "D:/Optimization-Research/UNSW_NB15/data/raw/UNSW-NB15_1.csv"
    features_info_path = "D:/Optimization-Research/UNSW_NB15/data/raw/NUSW-NB15_features.csv"
    processed_dir = "D:/Optimization-Research/UNSW_NB15/data/processed"
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load feature information to get column names
    # Try different encodings to handle potential encoding issues
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    features_info = None
    
    for encoding in encodings:
        try:
            print(f"Trying to read features file with {encoding} encoding...")
            features_info = pd.read_csv(features_info_path, encoding=encoding)
            print(f"Successfully read features file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Could not read features file with {encoding} encoding")
            continue
    
    if features_info is None:
        raise ValueError("Could not read features file with any encoding")
    
    # Extract column names from the features info file
    column_names = features_info['Name'].tolist()
    
    # Load raw data with the column names
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path, header=None, names=column_names, low_memory=False)
    
    print(f"Raw data shape: {df.shape}")
    print(f"First few rows of raw data:")
    print(df.head())
    
    # Drop high-cardinality columns
    print("Dropping high-cardinality columns...")
    high_cardinality_cols = ['srcip', 'dstip', 'Stime', 'Ltime']
    df = drop_high_cardinality(df, high_cardinality_cols)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values per column:")
    print(missing_values[missing_values > 0])
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # If no explicit categorical columns, identify them based on the features info file
    if not categorical_cols:
        categorical_cols = features_info[features_info['Type'].str.strip().str.lower() == 'nominal']['Name'].tolist()
        # Remove high cardinality columns that were dropped
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Ensure target column is not in the feature lists
        if 'attack_cat' in categorical_cols:
            categorical_cols.remove('attack_cat')
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Prepare target column
    target_column = 'Label'
    
    # Handle categorical features using LabelEncoder for each column
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string first to handle any numeric values
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Encoded {col} with {len(le.classes_)} unique values")
    
    # Identify features that are likely to be skewed (byte counts, packet counts, etc.)
    skewed_features = ['sbytes', 'dbytes', 'Sload', 'Dload', 'Dpkts', 'Spkts', 'dur']
    skewed_features = [f for f in skewed_features if f in numeric_cols]
    
    # Split data into features and target
    X = df.drop(columns=[target_column, 'attack_cat'] if 'attack_cat' in df.columns else [target_column])
    y = df[target_column]
    
    # Split into train, validation, test (70%, 15%, 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Update numeric_cols to only include columns in X
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    
    # Create preprocessing pipeline
    # Numeric pipeline with standard log transformation instead of custom transformer
    # Skip log transformation for now due to pickling issues
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # Fit the preprocessor on the training data
    print("Fitting preprocessing pipeline on training data...")
    preprocessor.fit(X_train)
    
    # Transform the data
    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    preprocessor_path = os.path.join(processed_dir, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessing pipeline to {preprocessor_path}")
    
    # Combine the transformed features with the target for saving
    train_df = pd.DataFrame(X_train_transformed)
    train_df['Label'] = y_train.values
    
    val_df = pd.DataFrame(X_val_transformed)
    val_df['Label'] = y_val.values
    
    test_df = pd.DataFrame(X_test_transformed)
    test_df['Label'] = y_test.values
    
    # Save the processed datasets
    train_df.to_csv(os.path.join(processed_dir, "Training_dataset.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "Validation_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "Testing_dataset.csv"), index=False)
    
    print("Data preprocessing completed successfully.")
    print(f"Saved processed datasets to {processed_dir}")
    
    # Return the file paths for convenience
    return {
        "preprocessor": preprocessor_path,
        "train_data": os.path.join(processed_dir, "Training_dataset.csv"),
        "val_data": os.path.join(processed_dir, "Validation_dataset.csv"),
        "test_data": os.path.join(processed_dir, "Testing_dataset.csv")
    }

if __name__ == "__main__":
    preprocess_data()
