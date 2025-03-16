# Intrusion Detection System (IDS) Project

## Project Objectives

- **Data Preprocessing:**  
  - Clean, transform, and encode the UNSW-NB15 dataset using a custom preprocessing pipeline.
  - Handle numeric and categorical features appropriately.
  
- **Model Development:**  
  - Implement and compare multiple machine learning models for intrusion detection, including:
    - Decision Tree
    - Random Forest
    - XGBoost (with GPU acceleration)
    - Artificial Neural Networks (ANN)
    - Convolutional Neural Networks (CNN)
    - Recurrent Neural Networks (LSTM)
  
- **Hyperparameter Optimization:**  
  - Utilize Bayesian Optimization (via scikit‑optimize) to fine-tune model parameters for improved performance.

- **Model Evaluation:**  
  - Evaluate models using key performance metrics: accuracy, precision, recall, F1-score, AUC-ROC, balanced accuracy, and confusion matrix analysis.
  - Perform cross‑validation and manual inspection of predictions to check for overfitting and ensure reliable performance.

- **End-to-End Workflow:**  
  - Ensure a robust workflow from raw data loading and preprocessing to model training, evaluation, and saving performance metrics for comparison.

## Setup and Installation

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

**Windows:**
```bash
python -m venv environment
.\environment\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv environment
source environment/bin/activate
```

### 3. Install Required Libraries

Install the necessary libraries using pip. You can either install them individually or use the provided `requirements.txt` file.

**To install using pip:**

```bash
pip install pandas numpy joblib scikit-learn scikit-optimize xgboost tensorflow
```

**Alternatively, if a `requirements.txt` file is provided, run:**

```bash
pip install -r requirements.txt
```

### 4. Dataset

Download the UNSW-NB15 dataset from its official source and place the training and testing CSV files in the appropriate data directory (e.g., `data/row/`).

### 5. Running the Scripts

Each model is implemented in its own script. For example:

- **Decision Tree:** `DT.py`
- **Random Forest:** `RF.py`
- **XGBoost:** `XGBoost.py`
- **LSTM:** `LSTM.py`
- **CNN:** `CNN.py`
- **ANN:** `ANN.py`

To run a script, simply execute:

```bash
python <script_name>.py
```

For example, to run the XGBoost model:

```bash
python XGBoost.py
```

## Additional Notes

- **Preprocessing Pipeline:**  
  The preprocessing functions are defined in `preprocessing_functions.py` and a pipeline is built and saved for reuse across different models.

- **GPU Acceleration:**  
  If you plan to use GPU acceleration (e.g., with XGBoost or TensorFlow), ensure that your system has an NVIDIA GPU with CUDA and cuDNN installed. Use the command `nvidia-smi` to verify your GPU status.

- **VSCode Issues:**  
  If VSCode doesn’t recognize certain libraries (like `tensorflow.keras`), ensure that the correct Python interpreter is selected, and consider adding the virtual environment’s `Lib\site-packages` directory to your VSCode settings.

