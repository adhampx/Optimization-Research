import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Compare the results of all implemented models for the UNSW-NB15 IDS dataset.
    
    This script reads the results from all CSV files in the Results directory,
    then creates comparison tables and visualizations for the key metrics.
    """
    # Path to the results directory
    results_dir = "D:/Optimization-Research/UNSW_NB15/src/Basic_Models/Results"
    
    # Get all CSV files in the results directory
    result_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files: {[os.path.basename(f) for f in result_files]}")
    
    # Initialize a list to store all results
    all_results = []
    
    # Read all result files
    for file_path in result_files:
        try:
            model_results = pd.read_csv(file_path)
            # If there are multiple runs for the same model, take the most recent one
            model_results = model_results.sort_values('Timestamp', ascending=False).iloc[0:1]
            all_results.append(model_results)
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
    
    # Combine all results into a single DataFrame
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        print(f"Combined results shape: {combined_results.shape}")
    else:
        print("No valid results found.")
        return
    
    # Select and reorganize key metrics
    key_metrics = [
        'Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 
        'Test_F1_Score', 'Test_AUC_ROC', 'Test_FPR', 'Test_FRR'
    ]
    
    # Check which metrics are available in the results
    available_metrics = [m for m in key_metrics if m in combined_results.columns]
    
    if 'Model' not in available_metrics:
        print("Model column not found in results. Cannot proceed with comparison.")
        return
    
    metrics_df = combined_results[available_metrics].copy()
    
    # Display the results table
    print("\nModel Comparison Results:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(metrics_df)
    
    # Create a nice formatted table for the report
    if len(metrics_df) > 0:
        # Format the metrics to display percentages with 2 decimal places
        for col in metrics_df.columns:
            if col != 'Model' and col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
        
        # Save the formatted table to CSV
        formatted_table_path = os.path.join(results_dir, "model_comparison.csv")
        metrics_df.to_csv(formatted_table_path, index=False)
        print(f"\nFormatted comparison table saved to: {formatted_table_path}")
    
    # Create visualizations for key metrics
    if len(combined_results) > 1:
        create_visualizations(combined_results, results_dir)

def create_visualizations(combined_results, results_dir):
    """
    Create visualizations to compare model performance.
    
    Args:
        combined_results (pd.DataFrame): DataFrame containing results for all models.
        results_dir (str): Directory to save the visualizations.
    """
    print("\nCreating performance comparison visualizations...")
    
    # Set up the visualizations directory
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Ensure we have the model column
    if 'Model' not in combined_results.columns:
        print("Model column not found in results. Cannot create visualizations.")
        return
    
    # Define the metrics to visualize
    performance_metrics = [
        'Test_Accuracy', 'Test_Precision', 'Test_Recall', 
        'Test_F1_Score', 'Test_AUC_ROC'
    ]
    
    error_metrics = [
        'Test_FPR', 'Test_FRR'
    ]
    
    # Check which metrics are available
    available_perf_metrics = [m for m in performance_metrics if m in combined_results.columns]
    available_error_metrics = [m for m in error_metrics if m in combined_results.columns]
    
    if not available_perf_metrics and not available_error_metrics:
        print("No metrics available for visualization.")
        return
    
    # Create performance metrics visualization
    if available_perf_metrics:
        plt.figure(figsize=(12, 8))
        
        # Extract model names and performance metrics
        models = combined_results['Model'].tolist()
        
        # Set up a bar chart
        x = np.arange(len(models))
        width = 0.15  # Width of the bars
        multiplier = 0
        
        # Plot each metric as a group of bars
        for metric in available_perf_metrics:
            values = combined_results[metric].tolist()
            offset = width * multiplier
            rects = plt.bar(x + offset, values, width, label=metric.replace('Test_', ''))
            multiplier += 1
        
        # Set up the chart
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x + width * (len(available_perf_metrics) - 1) / 2, models)
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of each bar
        for ax in plt.gcf().axes:
            for cont in ax.containers:
                for rect in cont:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Save the figure
        performance_fig_path = os.path.join(viz_dir, "performance_metrics.png")
        plt.tight_layout()
        plt.savefig(performance_fig_path)
        plt.close()
        print(f"Performance metrics visualization saved to: {performance_fig_path}")
    
    # Create error metrics visualization
    if available_error_metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract model names and error metrics
        models = combined_results['Model'].tolist()
        
        # Set up a bar chart
        x = np.arange(len(models))
        width = 0.35  # Width of the bars
        
        # Plot each error metric
        for i, metric in enumerate(available_error_metrics):
            values = combined_results[metric].tolist()
            offset = width * i
            rects = plt.bar(x + offset - width/2 + i*width, values, width, label=metric.replace('Test_', ''))
        
        # Set up the chart
        plt.ylabel('Error Rate')
        plt.title('Error Metrics Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of each bar
        for ax in plt.gcf().axes:
            for cont in ax.containers:
                for rect in cont:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.001,
                            f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Save the figure
        error_fig_path = os.path.join(viz_dir, "error_metrics.png")
        plt.tight_layout()
        plt.savefig(error_fig_path)
        plt.close()
        print(f"Error metrics visualization saved to: {error_fig_path}")
    
    # Create a radar chart for model comparison
    if len(available_perf_metrics) >= 3:
        plt.figure(figsize=(10, 10))
        
        # Set up the radar chart
        models = combined_results['Model'].tolist()
        metrics = available_perf_metrics
        
        # Calculate angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Set up the plot
        ax = plt.subplot(111, polar=True)
        
        # Add each model as a line in the radar chart
        for i, model in enumerate(models):
            values = combined_results.loc[combined_results['Model'] == model, metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels for each metric
        plt.xticks(angles[:-1], [m.replace('Test_', '') for m in metrics])
        
        # Set y limits to [0, 1] for better visualization
        plt.ylim(0.5, 1.0)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        plt.title('Model Performance Comparison (Radar Chart)')
        
        # Save the figure
        radar_fig_path = os.path.join(viz_dir, "model_radar_chart.png")
        plt.tight_layout()
        plt.savefig(radar_fig_path)
        plt.close()
        print(f"Radar chart visualization saved to: {radar_fig_path}")

if __name__ == "__main__":
    main() 