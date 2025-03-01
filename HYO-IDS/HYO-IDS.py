import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipaddress  # for IP address conversion

# Data preprocessing & balancing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# For pseudo-label generation (not used for UNSW-NB15 in this context)
from sklearn.ensemble import IsolationForest

# Model evaluation and cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)
from xgboost import XGBClassifier

# Bayesian Optimization for hyperparameter tuning
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Statistical tests (if needed)
from scipy.stats import friedmanchisquare

# For post-hoc analysis (if needed)
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None
    print("Warning: scikit-posthocs not installed. Install it if post-hoc testing is required.")

# For SHAP analysis
import shap

# ----------------------------
# Data Preparation Function (Per File)
# ----------------------------
def prepare_data_df(df, target_col='label'):
    """
    Prepares the UNSW-NB15 dataset from a DataFrame.
    Expects exactly 49 columns:
      - Uses the first 47 columns as features.
      - Discards the 48th column (attack-type string).
      - Uses the 49th column (boolean) as the label.
    The function:
      - Removes duplicates.
      - Fills missing values using forward and backward fill.
      - Converts the first 47 columns to numeric. For object-type columns,
        it first attempts to convert directly to float. If that fails, it tries
        to interpret the value as an IPv4 address and converts it to an integer.
        Any values that still cannot be converted are set to NaN and then filled
        with the column median (or 0 if the median is NaN).
      - Normalizes the features using StandardScaler.
      - Balances the dataset using SMOTE.
    """
    # Ensure the dataframe has 49 columns
    if df.shape[1] != 49:
        raise ValueError(f"Expected 49 columns but got {df.shape[1]}")
    
    df = df.copy()
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Fill missing values using forward then backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # For UNSW-NB15:
    # Features: columns 0 to 46 (first 47 columns)
    # Label: column 48 (49th column)
    X = df.iloc[:, :47]
    y = df.iloc[:, 48]
    
    # Convert label to int (0/1)
    y = y.astype(int)
    
    # Define a helper function to attempt conversion
    def try_convert(val):
        try:
            return float(val)
        except Exception:
            try:
                return int(ipaddress.IPv4Address(val))
            except Exception:
                return np.nan

    # For each column in X, if its dtype is object, attempt conversion
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(try_convert)
        else:
            # Also ensure numeric columns are float type
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any NaN values: if the column's median is not available (all NaN), fill with 0
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Balance data using SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE, class distribution: {np.bincount(y_bal)}")
    
    return X_bal, y_bal, list(X.columns)

# ----------------------------
# Hybrid Optimization Functions
# ----------------------------
def initialize_population(num_candidates, num_features):
    """
    Initializes candidate feature subsets as binary vectors.
    Uses Opposition-Based Learning (OBL) for diversity.
    """
    population = []
    for _ in range(num_candidates):
        candidate = np.random.randint(0, 2, num_features)
        if np.sum(candidate) == 0:
            candidate[np.random.randint(0, num_features)] = 1
        population.append(candidate)
        # Create the opposite candidate
        opposite = 1 - candidate
        if np.sum(opposite) == 0:
            opposite[np.random.randint(0, num_features)] = 1
        population.append(opposite)
    unique_pop = []
    seen = set()
    for cand in population:
        tup = tuple(cand)
        if tup not in seen:
            seen.add(tup)
            unique_pop.append(cand)
    return unique_pop[:num_candidates]

def levy_flight(candidate, beta=1.5):
    """
    Applies a Levy flight-inspired update by flipping a few bits.
    """
    num_features = len(candidate)
    n_flip = np.random.poisson(1)
    flip_indices = np.random.choice(num_features, size=min(n_flip, num_features), replace=False)
    new_candidate = candidate.copy()
    new_candidate[flip_indices] = 1 - new_candidate[flip_indices]
    if np.sum(new_candidate) == 0:
        new_candidate[np.random.randint(0, num_features)] = 1
    return new_candidate

def whale_rime_update(candidate, best_candidate, iteration, max_iterations):
    """
    Updates a candidate using a simulated Whale-Rime strategy:
      - Moves the candidate toward the best candidate with probability α (decreasing over iterations).
      - Adds random perturbations.
    """
    num_features = len(candidate)
    new_candidate = candidate.copy()
    alpha = 2 * (1 - iteration / max_iterations)
    for i in range(num_features):
        if np.random.rand() < alpha:
            new_candidate[i] = best_candidate[i]
        else:
            if np.random.rand() < 0.1:
                new_candidate[i] = 1 - new_candidate[i]
    if np.sum(new_candidate) == 0:
        new_candidate[np.random.randint(0, num_features)] = 1
    return new_candidate

def evaluate_candidate(candidate, X, y, delta=0.5, epsilon=0.5, cv_folds=3):
    """
    Evaluates a candidate feature subset using cross-validation with a GPU-accelerated XGBoost classifier.
    Fitness = AUC - delta * FPR - epsilon * (|Selected| / |Total|)
    """
    if np.sum(candidate) == 0:
        return -np.inf

    selected_features = np.where(candidate == 1)[0]
    X_subset = X.iloc[:, selected_features]
    
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                        verbosity=0, random_state=42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    auc_scores = []
    fpr_list = []
    for train_idx, test_idx in cv.split(X_subset, y):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        auc_scores.append(auc)
        
        y_pred = (y_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_list.append(fpr)
    
    avg_auc = np.mean(auc_scores)
    avg_fpr = np.mean(fpr_list)
    feature_ratio = np.sum(candidate) / X.shape[1]
    fitness = avg_auc - delta * avg_fpr - epsilon * feature_ratio
    return fitness

def hybrid_whale_rime_optimization(X, y, delta=0.5, epsilon=0.5, num_candidates=10, max_iterations=20):
    """
    Runs the hybrid optimization process:
      - Initializes candidate feature subsets.
      - Iteratively updates candidates using the Whale-Rime strategy, Levy flights, and periodic opposition-based learning.
      - Returns the best candidate (binary mask) and its fitness.
    """
    num_features = X.shape[1]
    population = initialize_population(num_candidates, num_features)
    best_candidate = None
    best_fitness = -np.inf

    for iteration in range(max_iterations):
        print(f"Optimization iteration {iteration+1}/{max_iterations}")
        for candidate in population:
            fit = evaluate_candidate(candidate, X, y, delta, epsilon)
            if fit > best_fitness:
                best_fitness = fit
                best_candidate = candidate.copy()
        
        new_population = []
        for candidate in population:
            updated_candidate = whale_rime_update(candidate, best_candidate, iteration, max_iterations)
            if np.random.rand() < 0.2:
                updated_candidate = levy_flight(updated_candidate)
            if iteration % 5 == 0:
                obl_candidate = 1 - updated_candidate
                if evaluate_candidate(obl_candidate, X, y, delta, epsilon) > evaluate_candidate(updated_candidate, X, y, delta, epsilon):
                    updated_candidate = obl_candidate
            new_population.append(updated_candidate)
        population = new_population

    print("Optimization complete. Best fitness:", best_fitness)
    return best_candidate, best_fitness

# ----------------------------
# Classifier Evaluation & Hyperparameter Tuning (GPU Enabled)
# ----------------------------
def bayesian_hyperparameter_tuning(X, y, selected_features, cv_folds=3, n_iter=25):
    """
    Tunes XGBoost hyperparameters using Bayesian Optimization on the selected feature subset.
    GPU acceleration is enabled.
    Returns the best XGBoost model.
    """
    X_subset = X.iloc[:, selected_features]
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                        verbosity=0, random_state=42)
    param_space = {
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='uniform'),
        'n_estimators': Integer(50, 300),
        'subsample': Real(0.5, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.5, 1.0, prior='uniform')
    }
    
    bayes_cv = BayesSearchCV(estimator=xgb,
                             search_spaces=param_space,
                             cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                             n_iter=n_iter,
                             scoring='roc_auc',
                             random_state=42,
                             n_jobs=-1)
    bayes_cv.fit(X_subset, y)
    print("Best hyperparameters:", bayes_cv.best_params_)
    return bayes_cv.best_estimator_

# ----------------------------
# SHAP Analysis
# ----------------------------
def shap_analysis(model, X, y, file_id=1, output_dir="shap_outputs"):
    """
    Computes SHAP values and saves a global summary plot and a local force plot.
    The output files are fixed so that each run produces one set of SHAP outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_sample = X.iloc[:100, :]  # Sample for efficiency

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    summary_path = os.path.join(output_dir, "HYO-IDS\\shap_summary_file1.png")
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    # Local force plot for the first instance; saved as HTML.
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X_sample.iloc[0, :], matplotlib=False)
    force_path = os.path.join(output_dir, "HYO-IDS\\shap_force_file1.html")
    shap.save_html(force_path, force_plot)
    print(f"SHAP plots saved: {summary_path} and {force_path}")
    
    return summary_path, force_path

# ----------------------------
# Evaluation Metrics
# ----------------------------
def compute_metrics(model, X, y, selected_features):
    """
    Computes evaluation metrics (Accuracy, Precision, Recall, F1-score, FPR, AUC,
    and Feature Reduction Rate) for the model on the given feature subset.
    """
    X_subset = X.iloc[:, selected_features]
    y_pred = model.predict(X_subset)
    y_proba = model.predict_proba(X_subset)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    total_features = X.shape[1]
    num_selected = len(selected_features)
    feature_reduction_rate = (total_features - num_selected) / total_features
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'FPR': fpr,
        'AUC': auc,
        'Feature_Reduction_Rate': feature_reduction_rate
    }

# ----------------------------
# Main Execution: Process Files as Stream & Aggregate Runs
# ----------------------------
def main():
    # List of file paths (each file contains 49 columns as per UNSW-NB15)
    file_paths = [
        r"DataSets\\UNSW-NB15\\UNSW-NB15_1.csv",
        r"DataSets\\UNSW-NB15\\UNSW-NB15_2.csv",
        r"DataSets\\UNSW-NB15\\UNSW-NB15_3.csv",
        r"DataSets\\UNSW-NB15\\UNSW-NB15_4.csv"
    ]
    target_column = "label"  # The label is in the 49th column
    delta = 0.5
    epsilon = 0.5
    num_candidates = 10
    max_iterations = 20
    cv_folds = 3
    bayes_iter = 25

    # Number of runs for the entire pipeline
    num_runs = 199
    
    # Path for the aggregated report CSV
    report_path = "HYO-IDS\\final_aggregated_report.csv"
    # Create the report file with header if it doesn't exist
    if not os.path.exists(report_path):
        with open(report_path, 'w') as f:
            f.write("Run,Accuracy,Precision,Recall,F1-score,FPR,AUC,Feature_Reduction_Rate,Training_Time_sec\n")
    
    # Execute the pipeline for the specified number of runs
    for run in range(1, num_runs + 1):
        print(f"\n=== Executing Run {run} ===")
        run_start = time.time()
        run_metrics = []
        
        # Process each file individually (stream processing)
        for file in file_paths:
            print(f"\nProcessing file: {file}")
            df = pd.read_csv(file, low_memory=False)
            # Prepare data for this file (yields exactly 47 features and a label)
            X, y, feature_names = prepare_data_df(df, target_col=target_column)
            
            # Hybrid feature selection on this file
            best_candidate, best_fit = hybrid_whale_rime_optimization(X, y, delta, epsilon, num_candidates, max_iterations)
            selected_features = np.where(best_candidate == 1)[0]
            print(f"Optimal feature subset (indices): {selected_features}")
            print(f"Number of selected features: {len(selected_features)} out of {X.shape[1]}")
            
            # Hyperparameter tuning on the selected features
            best_model = bayesian_hyperparameter_tuning(X, y, selected_features, cv_folds=cv_folds, n_iter=bayes_iter)
            
            # Evaluate the model on this file
            file_metrics = compute_metrics(best_model, X, y, selected_features)
            run_metrics.append(file_metrics)
            
            # Perform SHAP analysis (outputs overwritten per file; final run's outputs persist)
            shap_summary, shap_force = shap_analysis(best_model, X.iloc[:, selected_features], y, file_id=1)
        
        # Average the metrics across all files for this run
        avg_metrics = { key: np.mean([m[key] for m in run_metrics]) for key in run_metrics[0].keys() }
        run_time = time.time() - run_start
        avg_metrics['Training_Time_sec'] = run_time
        print(f"\nRun {run} Aggregated Metrics: {avg_metrics}")
        
        # Append the run's aggregated metrics to the CSV report
        with open(report_path, 'a') as f:
            row = (f"Run{run},{avg_metrics['Accuracy']},{avg_metrics['Precision']},"
                   f"{avg_metrics['Recall']},{avg_metrics['F1-score']},{avg_metrics['FPR']},"
                   f"{avg_metrics['AUC']},{avg_metrics['Feature_Reduction_Rate']},{avg_metrics['Training_Time_sec']}\n")
            f.write(row)
        print(f"Run {run} completed in {run_time:.2f} sec; results appended to {report_path}")
    
    print(f"\nFinal aggregated report is available in {report_path}")
    print("Note: The SHAP output files (shap_summary_file1.png and shap_force_file1.html) reflect the final run's analysis.")

if __name__ == "__main__":
    main()
