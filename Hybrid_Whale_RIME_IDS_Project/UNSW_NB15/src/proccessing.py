import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Define a function for log transformation using log1p (to handle zero values)
def log_transform(values):
    return np.log1p(values)

# Function to selectively apply log transformation on the column corresponding to 'dbytes'
# X is expected to be a NumPy array and feature_names is a list of column names in order.
def select_and_log_transform(X, feature_names):
    X_transformed = X.copy()
    if 'dbytes' in feature_names:
        idx = feature_names.index('dbytes')
        X_transformed[:, idx] = log_transform(X_transformed[:, idx])
    return X_transformed

# Named function (at module level) to be used with FunctionTransformer.
# It uses the global variable 'numeric_cols' to determine the column order.
def apply_log_transform(X):
    return select_and_log_transform(X, numeric_cols)

# Function to drop high-cardinality columns (e.g., IP addresses)
def drop_high_cardinality(df):
    columns_to_drop = ['srcip', 'dstip'] if set(['srcip', 'dstip']).issubset(df.columns) else []
    return df.drop(columns=columns_to_drop, errors='ignore')

def main():
    # Load the training and test datasets
    df_train = pd.read_csv('D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\\data\\row\\UNSW_NB15_training-set.csv')
    df_test = pd.read_csv('D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\\data\\row\\UNSW_NB15_testing-set.csv')

    print("Training dataset preview:")
    print(df_train.head())

    # Drop high-cardinality columns from both datasets
    df_train = drop_high_cardinality(df_train)
    df_test = drop_high_cardinality(df_test)

    # Identify numeric columns from the training dataset
    numeric_cols_list = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    target_column = 'label'  # Adjust if your target column name is different

    # Remove the target column from the numeric features if present
    if target_column in numeric_cols_list:
        numeric_cols_list.remove(target_column)

    # Identify categorical columns from the training dataset
    categorical_cols_list = df_train.select_dtypes(include=['object']).columns.tolist()

    print("Numeric columns:", numeric_cols_list)
    print("Categorical columns:", categorical_cols_list)

    # Set the global variable 'numeric_cols' for use in apply_log_transform
    global numeric_cols
    numeric_cols = numeric_cols_list

    # Create the numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('log_transform', FunctionTransformer(apply_log_transform, validate=False)),
        ('scaler', MinMaxScaler())
    ])

    # Create the categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols_list),
        ('cat', categorical_pipeline, categorical_cols_list)
    ])

    # Prepare the target label: if non-numeric, apply label encoding
    if df_train[target_column].dtype == 'object':
        le = LabelEncoder()
        df_train[target_column] = le.fit_transform(df_train[target_column])
        if target_column in df_test.columns and df_test[target_column].dtype == 'object':
            df_test[target_column] = le.transform(df_test[target_column])
        # Optionally, you can save the label encoder as well:
        # joblib.dump(le, 'label_encoder.joblib')
        print("Label encoding applied on training dataset.")
    else:
        print("Target column is already numeric in training dataset.")

    # Separate features and target labels for training and test datasets
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    if target_column in df_test.columns:
        X_test = df_test.drop(columns=[target_column])
    else:
        X_test = df_test.copy()

    # Fit the preprocessor on the training data and transform both training and test features
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("Preprocessing complete for both training and test datasets.")

    # Save the preprocessing pipeline as a joblib file for later reuse during inference.
    joblib.dump(preprocessor, 'D:\\Optimization-Research\\Hybrid_Whale_RIME_IDS_Project\\UNSW_NB15\\data\\proccessed\\preprocessing_pipeline.joblib')
    print("Preprocessing pipeline saved as 'preprocessing_pipeline.joblib'.")

if __name__ == '__main__':
    main()
