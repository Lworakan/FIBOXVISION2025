# Readme: RealSense Depth Calibration using Regression (PyCaret) - Tawan

## 1. Objective

To calibrate depth measurements from an Intel RealSense D450 camera using a regression model built with PyCaret (AutoML). The primary goal is to predict the true distance (`IntendedDistance_m`) based on the sensor's measured depth (`average_depth_m`), aiming for a prediction error of less than 2 cm (0.02 m), particularly for determining the 3D coordinates of a basketball hoop.

## 2. Data

- **Sensor:** Intel RealSense D450
- **Target Surface:** Wall (no specific object)
- **Distances:** Ranging from 4 meters to 9 meters.
- **Sampling:** Approximately 2000 depth measurements were collected for each distance increment.
- **Depth Measurement:** The average depth (`average_depth_m`) was calculated from region of interest (ROI) of the wall.

* **Locations:** Data was collected in at least two distinct environments:
  - `outdoor1stfloor`
  - `thirdfloor` (indoor)
* **Collected Data:** The final dataset (`combined_depth_data_for_automl.csv`) contains `average_depth_m`, `IntendedDistance_m` (ground truth), and `Location`.

## 3. Methodology

- **Tooling:** Python, Pandas, PyCaret.
- **Approach:** Calibration using Regression.
  - **Target Variable:** `IntendedDistance_m` (The true distance we want to predict).
  - **Primary Input Feature:** `average_depth_m` (The depth value measured by the RealSense sensor).
  - **Additional Input Feature:** `Location` (Categorical feature representing the environment).
- **Feature Engineering (PyCaret `setup`):**
  - `polynomial_features=True`, `polynomial_degree=2`: Automatically creates `average_depth_m^2` as an additional feature to help model non-linear sensor error characteristics.
  - One-Hot Encoding was applied to the `Location` feature.
- **AutoML Process (PyCaret):**
  1.  **`setup`:** Configured the PyCaret environment, defining the target, features (numeric and categorical), and enabling polynomial features. Data was split into training (70%) and test (30%) sets.
  2.  **`compare_models`:** Automatically trained and evaluated various regression models using 10-fold cross-validation on the training set, sorted by Root Mean Squared Error (RMSE). LightGBM, Gradient Boosting Regressor (GBR), and XGBoost consistently performed best. LightGBM was initially selected.
  3.  **`tune_model`:** Performed hyperparameter tuning on the best model (LightGBM) using 50 iterations (`n_iter=50`) to optimize for RMSE. This resulted in a slightly improved tuned LightGBM model.
  4.  **`blend_models`:** Attempted to blend the top models, but this step was skipped due to insufficient diversity/minor issues in model selection for blending in the final run.
  5.  **Model Selection:** The tuned LightGBM model (`tuned_lgbm_loc`) was selected as the final model based on cross-validation performance.
  6.  **`predict_model`:** Evaluated the performance of the final selected model on the unseen **Test Set**.
  7.  **Error Analysis:** Calculated and analyzed RMSE and Mean Absolute Error (MAE) specifically for each `Location` on the test set results.

## 4. Results

### 4.1. Best Model

The best performing model identified and selected after tuning was **Light Gradient Boosting Machine (LGBM)** (`tuned_lgbm_loc`).

### 4.2. Overall Performance (on Test Set)

Metrics calculated by `predict_model` on the unseen test data for the final tuned LGBM model:

- **RMSE:** 0.2525 meters (25.3 cm)
- **MAE:** 0.1091 meters (10.9 cm)
- **R2:** 0.9783

### 4.3. Location-Specific Performance (on Test Set)

This analysis revealed a significant difference in performance between the two locations:

- **`outdoor1stfloor`:**
  - **RMSE:** 0.3558 meters (35.6 cm)
  - **MAE:** 0.2113 meters (21.1 cm)
  - _Interpretation:_ Performance is relatively poor with high variability (error standard deviation was also calculated to be high, ~35.6 cm). The model struggles to consistently calibrate the sensor readings accurately in this environment.

* **`thirdfloor` (Indoor):**
  - **RMSE:** 0.0119 meters (1.19 cm)
  - **MAE:** 0.0056 meters (0.56 cm)
  - _Interpretation:_ Performance is **excellent** and **meets the < 2 cm error target**. The model successfully calibrates the sensor readings with high accuracy and low variability (error standard deviation was ~1.2 cm) in this environment.

## 5. Analysis & Conclusion

1.  **Calibration Approach is Valid:** Predicting the true distance (`IntendedDistance_m`) from the sensor reading (`average_depth_m`) using regression (especially with polynomial features) is a much more effective approach for sensor calibration compared to predicting the sensor reading from the true distance. This significantly reduced the overall error compared to initial attempts.
2.  **Location is Crucial:** The environment (`Location`) has a major impact on the RealSense sensor's accuracy and/or the model's ability to calibrate it.
3.  **Goal Achieved (Conditionally):** The target error of < 2 cm **was achieved** for the `thirdfloor` (indoor) environment, with an impressive RMSE of 1.19 cm and MAE of 0.56 cm on the test set.
4.  **Outdoor Challenges:** The model's performance at `outdoor1stfloor` is significantly worse (RMSE ~35.6 cm, MAE ~21.1 cm) and does not meet the target. This suggests that factors present in the outdoor environment (e.g., lighting variations, different surface properties at range, sensor noise characteristics) significantly affect the depth readings in a way that the current model and features (`average_depth_m`, `Location`, polynomial terms) cannot fully compensate for.
5.  **Feature Limitation:** The high error variance outdoors suggests that additional features might be needed to capture the conditions influencing the sensor's inaccuracy in that specific environment. IMU data was considered but not usable due to dataset incompatibility.

## 6. Recommendations & Next Steps

1.  **Investigate `outdoor1stfloor` Data/Environment:** Analyze the raw data specifically for `outdoor1stfloor`. Are there significant outliers? Plot the raw `average_depth_m` vs `IntendedDistance_m` to visualize the noise level. What environmental factors could be contributing to the higher error and variance (sunlight, surface reflectivity, temperature)?
2.  **Explore Additional Features (If Possible):** Revisit if *any* other relevant data can be extracted alongside depth from the RealSense SDK for future data collection (e.g., confidence score, standard deviation of depth within the ROI). These could significantly improve model performance, especially outdoors.
3.  **Model Transferability Test:** Proceed with the planned test to see how the current model (trained on both locations) performs when applied *only* to `thirdfloor` data and *only* to `outdoor1stfloor` data without retraining. This will further quantify the impact of location. The expectation is it will perform well on `thirdfloor` data but poorly on `outdoor1stfloor` data.
4.  **Consider LiDAR:** Given the excellent indoor results but poor outdoor results relative to the strict < 2 cm requirement, and the difficulty in obtaining further relevant features for the RealSense model, switching to LiDAR for scenarios requiring high accuracy in diverse or challenging (outdoor) environments remains a strong consideration, as planned.
5.  **Refine Indoor Model (Optional):** Although the indoor performance is already excellent, minor improvements might be possible by tuning specifically on indoor data or exploring slight variations in polynomial degree or model types (though LightGBM is already very strong).