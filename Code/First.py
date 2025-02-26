import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data preprocessing & balancing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# For pseudo-label generation when target column is missing
from sklearn.ensemble import IsolationForest

# Model evaluation and cross validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_curve)
from xgboost import XGBClassifier

# Bayesian Optimization for hyperparameter tuning
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Statistical tests
from scipy.stats import friedmanchisquare

# For post-hoc Nemenyi test (install via pip install scikit-posthocs)
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None
    print("Warning: scikit-posthocs not installed. Please install it for Nemenyi analysis.")

# For SHAP analysis
import shap

# ----------------------------
# 1. Data Preparation Functions
# ----------------------------
def prepare_data(filepath, target_col='label'):
    """
    Loads and preprocesses a dataset:
      - Loads CSV with low_memory disabled.
      - Removes duplicates and fills missing values using forward and backward fill.
      - If the target column is not present, generates pseudo-labels using IsolationForest.
        For pseudo-labeling, only numeric data is used. If the numeric subset is empty,
        attempts to convert object columns to numeric.
      - Retains only numeric features for normalization and further analysis.
      - Normalizes features using StandardScaler.
      - Balances classes using SMOTE.
    """
    # Load dataset with low_memory disabled
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {filepath} with shape {df.shape}")
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values: fill forward then backward
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # For pseudo-labeling, select numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # If numeric data is empty, try converting all columns to numeric
    if df_numeric.empty:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df_numeric = df.select_dtypes(include=[np.number])
    
    # If target column is not present, generate pseudo-labels using IsolationForest
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Generating pseudo-labels using IsolationForest.")
        if df_numeric.empty or df_numeric.shape[0] == 0:
            raise ValueError("No numeric data available for pseudo-label generation!")
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(df_numeric)
        # Map anomalies (-1) to 1 (attack) and normal instances to 0
        df[target_col] = (preds == -1).astype(int)
    
    # After labeling, retain only numeric features (excluding the target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    X = df[numeric_cols]
    y = df[target_col]
    
    # Feature normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Data balancing with SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE, class distribution: {np.bincount(y_bal)}")
    
    return X_bal, y_bal, X.columns.tolist()

# ----------------------------
# 2. Hybrid Optimization Framework Functions
# ----------------------------
def initialize_population(num_candidates, num_features):
    """
    Initializes a population of candidate feature subsets as binary vectors.
    Uses Opposition-Based Learning (OBL) for diversity.
    """
    population = []
    for _ in range(num_candidates):
        candidate = np.random.randint(0, 2, num_features)
        if np.sum(candidate) == 0:
            candidate[np.random.randint(0, num_features)] = 1
        population.append(candidate)
        # Generate the opposite candidate (OBL)
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
      - Moves candidate toward the best candidate with a probability that decreases over iterations.
      - Adds random perturbations (rime-inspired) and occasional bit flips.
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
    Evaluates a candidate feature subset using cross-validation with an XGBoost classifier.
    The fitness score is computed as: F = AUC - delta * FPR - epsilon * (|S| / |F|)
    """
    if np.sum(candidate) == 0:
        return -np.inf

    selected_features = np.where(candidate == 1)[0]
    X_subset = X.iloc[:, selected_features]
    
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42)
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
    Implements the hybrid optimization process:
      - Initializes candidate feature subsets with OBL.
      - Iteratively updates candidates using the Whale-Rime strategy, Levy flights, and periodic OBL.
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
# 3. Wrapper-Based Classifier Evaluation & Bayesian Hyperparameter Tuning
# ----------------------------
def bayesian_hyperparameter_tuning(X, y, selected_features, cv_folds=3, n_iter=25):
    """
    Tunes XGBoost hyperparameters using Bayesian Optimization on the selected feature subset.
    Returns the best model found.
    """
    X_subset = X.iloc[:, selected_features]
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42)
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
# 4. SHAP Analysis Functions
# ----------------------------
def shap_analysis(model, X, y, file_id, output_dir="shap_outputs"):
    """
    Computes SHAP values and saves global summary and local force plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_sample = X.iloc[:100, :]  # sample to reduce computation time

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    summary_path = os.path.join(output_dir, f"shap_summary_file{file_id}.png")
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    # Local force plot for the first instance; saved as HTML.
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X_sample.iloc[0, :], matplotlib=False)
    force_path = os.path.join(output_dir, f"shap_force_file{file_id}.html")
    shap.save_html(force_path, force_plot)
    print(f"SHAP plots saved for file {file_id}: {summary_path} and {force_path}")
    
    return summary_path, force_path

# ----------------------------
# 5. Evaluation Metrics and Statistical Testing
# ----------------------------
def compute_metrics(model, X, y, selected_features):
    """
    Computes evaluation metrics for the model on the selected features.
    Returns a dictionary of metrics including accuracy, precision, recall, F1, FPR, AUC, and feature reduction rate.
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

def perform_statistical_tests(results_list, metric='AUC'):
    """
    Performs a Friedman test across experiments (files) for the specified metric.
    If significant and scikit-posthocs is available, performs a post-hoc Nemenyi analysis.
    """
    metric_values = [res[metric] for res in results_list]
    stat, p_value = friedmanchisquare(*[[val] for val in metric_values])
    print(f"Friedman test for {metric}: statistic={stat:.4f}, p-value={p_value:.4f}")
    
    nemenyi_results = None
    if sp is not None and len(metric_values) > 1:
        df = pd.DataFrame({'File': np.arange(len(metric_values)), metric: metric_values})
        df_long = pd.concat([df]*5, ignore_index=True)
        df_long['Rep'] = np.tile(np.arange(5), len(metric_values))
        nemenyi_results = sp.posthoc_nemenyi_friedman(df_long.pivot(index='Rep', columns='File', values=metric))
        print("Post-hoc Nemenyi analysis (p-values):")
        print(nemenyi_results)
    else:
        print("Not enough data for post-hoc Nemenyi analysis.")
    
    return stat, p_value, nemenyi_results

# ----------------------------
# 6. Main Workflow Execution
# ----------------------------
def main():
    # Define file paths for the four parts of the UNSW-NB15 dataset.
    file_paths = [
        r"DataSets\\UNSW-NB15\\UNSW-NB15_1.csv",
        r"DataSets\\UNSW-NB15\\UNSW-NB15_2.csv",
        r"DataSets\\UNSW-NB15\\UNSW-NB15_3.csv",
        r"DataSets\\UNSW-NB15\\UNSW-NB15_4.csv"
    ]
    target_column = "label"  # Expected target column name (pseudo-labels generated if missing)
    delta = 0.5
    epsilon = 0.5
    num_candidates = 10
    max_iterations = 20
    cv_folds = 3
    
    aggregated_results = []
    final_report = {}

    for file_id, filepath in enumerate(file_paths, start=1):
        print(f"\nProcessing file {file_id}: {filepath}")
        start_time = time.time()
        # Data Preparation: load, clean (fill missing), generate labels if needed, retain numeric features, normalize, and balance.
        X, y, feature_names = prepare_data(filepath, target_col=target_column)
        
        # Hybrid Optimization for Feature Selection
        best_candidate, best_fit = hybrid_whale_rime_optimization(X, y, delta, epsilon, num_candidates, max_iterations)
        selected_features = np.where(best_candidate == 1)[0]
        print(f"Optimal feature subset (indices): {selected_features}")
        print(f"Number of selected features: {len(selected_features)} out of {X.shape[1]}")
        
        # Bayesian Hyperparameter Tuning using the selected features
        best_model = bayesian_hyperparameter_tuning(X, y, selected_features, cv_folds=cv_folds, n_iter=25)
        
        # Evaluate classifier performance and compute metrics
        metrics = compute_metrics(best_model, X, y, selected_features)
        training_time = time.time() - start_time
        metrics['Training_Time_sec'] = training_time
        aggregated_results.append(metrics)
        
        # SHAP Analysis for model explainability
        shap_summary, shap_force = shap_analysis(best_model, X.iloc[:, selected_features], y, file_id)
        
        # Save file-specific report information
        final_report[f"File_{file_id}"] = {
            "Optimal_Features": selected_features.tolist(),
            "Best_Fitness": best_fit,
            "Hyperparameters": best_model.get_params(),
            "Metrics": metrics,
            "SHAP_Summary": shap_summary,
            "SHAP_Force": shap_force
        }
    
    # Print aggregated evaluation metrics for all files
    print("\nAggregated Evaluation Metrics:")
    for idx, res in enumerate(aggregated_results, start=1):
        print(f"File {idx}: {res}")
    
    # Perform statistical testing (using AUC as an example metric)
    friedman_stat, friedman_p, nemenyi_results = perform_statistical_tests(aggregated_results, metric='AUC')
    final_report["Statistical_Tests"] = {
        "Friedman_Statistic": friedman_stat,
        "Friedman_p_value": friedman_p,
        "Nemenyi": nemenyi_results.to_dict() if nemenyi_results is not None else "N/A"
    }
    
    # Save final aggregated report as CSV and JSON
    report_df = pd.DataFrame(aggregated_results, index=[f"File_{i}" for i in range(1, len(aggregated_results)+1)])
    report_path = "final_aggregated_report.csv"
    report_df.to_csv(report_path)
    print(f"\nFinal aggregated report saved to {report_path}")
    
    with open("final_report.json", "w") as f:
        json.dump(final_report, f, indent=4)
    print("Final detailed report saved to final_report.json")

if __name__ == "__main__":
    main()
