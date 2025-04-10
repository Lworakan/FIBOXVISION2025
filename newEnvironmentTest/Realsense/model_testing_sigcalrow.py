#!/usr/bin/env python3
"""
Prediction function for a single DataFrame row of RealSense detection data.
This can be imported and used in other scripts to make predictions on individual detections.
"""

import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model

# Path to the model
MODEL_PATH = '/home/lworakan/Documents/GitHub/FIBOXVISION2025/newEnvironmentTest/Realsense/model/final_calibrated_depth_model_outdoor'

# Global model variable to avoid reloading for each prediction
_model = None

def load_model_once():
    """Load the model if it hasn't been loaded yet"""
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return _model

def predict_ground_truth_from_row(row, location="Lab"):
    """
    Predict ground truth from a single DataFrame row
    
    Parameters:
    row - A pandas Series or dict containing detection data with 'Average_Depth_m'
    location - The location string (default: "Lab")
    
    Returns:
    float - The predicted ground truth value
    """
    try:
        # Load model (only happens once)
        model = load_model_once()
        
        # Extract depth value
        if isinstance(row, pd.Series):
            if 'Average_Depth_m' in row:
                depth_value = row['Average_Depth_m']
            else:
                raise KeyError("'Average_Depth_m' not found in input row")
        elif isinstance(row, dict):
            if 'Average_Depth_m' in row:
                depth_value = row['Average_Depth_m']
            else:
                raise KeyError("'Average_Depth_m' not found in input dictionary")
        else:
            raise TypeError("Input must be a pandas Series or dictionary")
        
        # Create input data for the model
        input_data = pd.DataFrame({
            'average_depth_m': [depth_value],
            'Location': [location]
        })
        
        # Make prediction
        prediction = predict_model(model, data=input_data)
        
        # Return the predicted value
        return prediction['prediction_label'].iloc[0]
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Example 1: Using a pandas Series
    print("\nExample 1: Using a pandas Series")
    example_series = pd.Series({
        'Timestamp': '2025-04-09 10:00:00',
        'Frame': 100,
        'X_min': 150, 'Y_min': 200, 'X_max': 250, 'Y_max': 300,
        'Confidence': 0.95,
        'Average_Depth_m': 4.25,
        'area': 10000
    })
    
    predicted_gt = predict_ground_truth_from_row(example_series)
    print(f"Input depth: {example_series['Average_Depth_m']}m")
    print(f"Predicted ground truth: {predicted_gt:.4f}m")
    
    # Example 2: Using a dictionary
    print("\nExample 2: Using a dictionary")
    example_dict = {
        'Timestamp': '2025-04-09 10:01:00',
        'Frame': 101,
        'X_min': 155, 'Y_min': 205, 'X_max': 255, 'Y_max': 305,
        'Confidence': 0.93,
        'Average_Depth_m': 5.10,
        'area': 10000
    }
    
    predicted_gt = predict_ground_truth_from_row(example_dict)
    print(f"Input depth: {example_dict['Average_Depth_m']}m")
    print(f"Predicted ground truth: {predicted_gt:.4f}m")
    
    # Example 3: Process multiple rows from a DataFrame
    print("\nExample 3: Processing multiple rows from a DataFrame")
    example_df = pd.DataFrame([
        {'Average_Depth_m': 3.5},
        {'Average_Depth_m': 4.0},
        {'Average_Depth_m': 4.5},
        {'Average_Depth_m': 5.0},
        {'Average_Depth_m': 5.5}
    ])
    
    # Add predictions column
    example_df['ground_truth'] = example_df.apply(
        lambda row: predict_ground_truth_from_row(row), axis=1
    )
    
    print(example_df)