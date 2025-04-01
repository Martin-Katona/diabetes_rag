import numpy as np
import pickle
import sys
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


MODEL_DIR = os.path.join(script_dir, "dia_saved_models")  # Directory where models are saved

# Define the dataset type (either "subset" or "all")
dataset_type = "subset"  # Change this to "all" if needed

# Construct the paths dynamically based on the dataset type
MINMAX_SCALER_PATH = os.path.join(MODEL_DIR, f'minmax_scaler_{dataset_type}.pkl')
STANDARD_SCALER_PATH = os.path.join(MODEL_DIR, f'standard_scaler_{dataset_type}.pkl')

# Construct the path to the model file
MODEL_PATH = os.path.join(script_dir, "dia_saved_models", f"dia_best_model_{dataset_type}.keras")
SHAP_EXPLAINER_PATH = os.path.join(script_dir, "dia_saved_models", f"dia_shap_explainer_{dataset_type}.bz2")
CONFIG_PATH = os.path.join(script_dir, "dia_saved_models", f"dia_best_model_dataset_config_{dataset_type}.pkl")

#MODEL_PATH = "diabetes/dia_saved_models/dia_best_model.keras"
# CONFIG_PATH = "diabetes/dia_saved_models/dia_best_model_dataset_config.pkl"
# SHAP_EXPLAINER_PATH = "diabetes/dia_saved_models/dia_shap_explainer.bz2"

# Load the model
model = load_model(MODEL_PATH)

# Load the dataset configuration
with open(CONFIG_PATH, "rb") as f:
    dataset_config = pickle.load(f)

# Expose configuration values
best_fold = dataset_config["best_fold"]
train_index = dataset_config["train_index"]
val_index = dataset_config["val_index"]
X_train = dataset_config["X_train"]
X_val = dataset_config["X_val"]
y_train = dataset_config["y_train"]
y_val = dataset_config["y_val"]

# Load the SHAP explainer
ex2 = joblib.load(SHAP_EXPLAINER_PATH)


# Load scalers
minmax_scaler = joblib.load(MINMAX_SCALER_PATH)
standard_scaler = joblib.load(STANDARD_SCALER_PATH)

import os
import joblib

def scale_input(input_dict):
    """
    Scales the input dictionary using fitted MinMaxScaler and StandardScaler.
    Ignores features with None values.

    Parameters:
        input_dict (dict): Dictionary containing feature values.

    Returns:
        dict: Scaled input dictionary.
    """
    # Define columns
    ordinal_cols = ['Age', 'GenHlth']
    numerical_cols = ['BMI', 'MentHlth', 'PhysHlth']

    # Load scalers
    # minmax_scaler = joblib.load(os.path.join(MODEL_DIR, 'minmax_scaler.pkl'))
    # standard_scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))

    minmax_scaler = joblib.load(MINMAX_SCALER_PATH)
    standard_scaler = joblib.load(STANDARD_SCALER_PATH)

    # Create a copy of the input dictionary to avoid modifying the original
    scaled_input = input_dict.copy()

    # Scale ordinal features
    if ordinal_cols:
        ordinal_features = [scaled_input[col] for col in ordinal_cols if col in scaled_input and scaled_input[col] is not None]
        if ordinal_features:
            scaled_ordinal = minmax_scaler.transform([ordinal_features])[0]
            for i, col in enumerate(ordinal_cols):
                if col in scaled_input and scaled_input[col] is not None:
                    scaled_input[col] = scaled_ordinal[i]

    # Scale numerical features
    if numerical_cols:
        numerical_features = [scaled_input[col] for col in numerical_cols if col in scaled_input and scaled_input[col] is not None]
        if numerical_features:
            scaled_numerical = standard_scaler.transform([numerical_features])[0]
            for i, col in enumerate(numerical_cols):
                if col in scaled_input and scaled_input[col] is not None:
                    scaled_input[col] = scaled_numerical[i]

    return scaled_input


def diabetes_predict(input_dict):
    """
    Predicts the probability of diabetes for a single person using provided features.

    Parameters:
        input_dict (dict): Dictionary containing feature values. Keys should match parameter names.

    Returns:
        float: Probability of diabetes (between 0 and 1).
    """
    # Get the number of features expected by the model
    num_features = model.input_shape[-1]

    # Prepare input features dynamically from the dictionary
    input_features = []
    
    for key, value in input_dict.items():
        if value is not None:  # Only include features with non-None values
            input_features.append(value)
    
    # Check if the number of provided features matches what the model expects
    if len(input_features) != num_features:
        raise ValueError(
            f"Model expects {num_features} features, but {len(input_features)} were provided."
        )

    # Convert to NumPy array and reshape for prediction
    input_features = np.array(input_features).reshape(1, -1)
    
    # Make a prediction
    prediction_probability = model.predict(input_features)[0][0]
    return prediction_probability


def explain_prediction(input_dict):
    """
    Explains the prediction using SHAP values and generates visualizations.

    Parameters:
        input_dict (dict): Dictionary containing feature values. Keys should match parameter names.

    Returns:
        shap_values: SHAP values for the input data.
    """
    # Get the number of features expected by the model
    num_features = model.input_shape[-1]

    # Prepare input features dynamically from the dictionary
    input_features = []
    
    for key, value in input_dict.items():
        if value is not None:  # Only include features with non-None values
            input_features.append(value)
    
    # Check if the number of provided features matches what the model expects
    if len(input_features) != num_features:
        raise ValueError(
            f"Model expects {num_features} features, but {len(input_features)} were provided."
        )

    # Convert to NumPy array and reshape for SHAP explanation
    input_features = np.array(input_features).reshape(1, -1)

    # Compute SHAP values
    shap_values = ex2(input_features)

    # Extract SHAP values for visualization
    shap_values_for_plot = shap_values[0]

    feature_names = dataset_config["X_val"].columns.tolist()
    shap_values_for_plot.feature_names = feature_names

   # Determine the save path based on execution environment
    if 'ipykernel' in sys.modules:
        # Running from Jupyter Notebook
        notebook_dir = os.getcwd()
        plot_dir = os.path.join(notebook_dir, 'diabetes', 'plots')
    else:
        # Running from terminal
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(script_dir, 'diabetes', 'plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Generate and save SHAP waterfall plot
    waterfall_plot_path = os.path.join(plot_dir, 'dia_waterfall_plot.png')
    shap.plots.waterfall(shap_values_for_plot, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_plot_path, dpi=300)
    plt.clf()

    # Generate and save SHAP bar plot
    bar_plot_path = os.path.join(plot_dir, 'dia_bar_plot.png')
    shap.plots.bar(shap_values_for_plot, show=False)
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300)
    plt.clf()

    return shap_values, shap_values_for_plot


