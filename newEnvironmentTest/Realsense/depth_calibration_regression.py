import pandas as pd
from pycaret.regression import *
import matplotlib.pyplot as plt # สำหรับ plot เพิ่มเติม (ถ้าต้องการ)
import numpy as np

# --- 1. โหลดและเตรียมข้อมูล ---
print("--- 1. Loading and Preparing Data ---")
# โหลดข้อมูลของคุณ (แก้ Path ถ้าจำเป็น)
csv_path = r'newEnvironmentTest\Realsense\collected_data\combined_depth_data_for_automl.csv'
df_full = pd.read_csv(csv_path)

# เลือกคอลัมน์สำหรับแนวทาง Calibration (ทำนายระยะจริงจากค่า Sensor) + Location
# Input Features: aver
# age_depth_m, Location
# Target Variable: IntendedDistance_m
data_for_setup = df_full[['average_depth_m', 'IntendedDistance_m', 'Location']]

print("Data prepared for PyCaret setup (with Location):")
print(data_for_setup.head())
print("\nData Info:")
data_for_setup.info() # ดูชนิดข้อมูลและค่า non-null

# (Optional) ลอง Plot ดูการกระจายตัวคร่าวๆ
plt.figure(figsize=(10, 6))
plt.scatter(data_for_setup['average_depth_m'], data_for_setup['IntendedDistance_m'], alpha=0.1, label='Data points')
plt.plot([4, 9], [4, 9], 'r--', label='Ideal Line (y=x)') # เส้นอ้างอิง
plt.xlabel("Average Depth Measured (m)")
plt.ylabel("Intended Distance (m)")
plt.title("Measured Depth vs Intended Distance")
plt.legend()
plt.grid(True)
plt.show()


# --- 2. Setup PyCaret Environment ---
print("\n--- 2. Setting up PyCaret Environment ---")
# session_id เพื่อให้ผลลัพธ์เหมือนเดิม
# html=False ถ้าใช้บน Colab
# เริ่มด้วย use_gpu=False ก่อน
reg_setup_calib_loc = setup(data=data_for_setup,
                              target='IntendedDistance_m',
                              session_id=456, # หรือใช้ ID ที่คุณต้องการ
                              numeric_features=['average_depth_m'],
                              categorical_features=['Location'],
                              polynomial_features=True, # สร้าง Feature พหุนาม
                              polynomial_degree=2,      # ดีกรี 2
                              # normalize=True, # ลองเปิด normalize ดู อาจช่วยบางโมเดล
                              log_experiment=False,
                              use_gpu=False,
                              html=False
                             )
# กด Enter เพื่อยืนยันตอนรัน setup


# --- 3. Compare Models ---
print("\n--- 3. Comparing Regression Models ---")
# เรียงตาม RMSE (น้อยดี)
best_calib_loc_model = compare_models(sort='RMSE')

print("\nBest calibration model (with Location) found by compare_models:")
print(best_calib_loc_model)


# --- 4. Hyperparameter Tuning (สำหรับโมเดลที่ดีที่สุดจาก compare_models) ---
print("\n--- 4. Tuning the Best Model ---")
tuned_best_model = None
try:
    print(f"Tuning {type(best_calib_loc_model).__name__} (n_iter=50)...")
    # ลอง tune ด้วย n_iter=50 (ปรับได้ตามเวลาที่มี)
    tuned_best_model = tune_model(best_calib_loc_model, optimize='RMSE', n_iter=50)
    print("\nTuned Model Results (Cross-Validation Metrics):")
    print(tuned_best_model)
except NameError:
    print("Error: 'compare_models' did not return a model or 'tune_model' is not defined.")
except Exception as e:
    print(f"An error occurred during tuning: {e}")

# --- 5. Ensemble Models (Blending) ---
# ลองรวมโมเดลดีๆ เข้าด้วยกัน (ตัวที่ Tune แล้ว + ตัวอื่นจาก compare_models)
print("\n--- 5. Blending Top Models ---")
blender_model = None
models_to_blend = []

# ใช้ตัวที่ Tune แล้วเป็นตัวหลัก (ถ้า Tune สำเร็จ) หรือใช้ตัว best เดิม
if tuned_best_model is not None:
    models_to_blend.append(tuned_best_model)
    base_model_for_comparison = tuned_best_model
elif 'best_calib_loc_model' in locals():
    models_to_blend.append(best_calib_loc_model)
    base_model_for_comparison = best_calib_loc_model
else:
    print("No base model available for blending.")
    base_model_for_comparison = None

# ลองเพิ่มโมเดลอื่นที่คะแนนดีรองๆ ลงมา (เช่น gbr, knn ถ้าไม่ได้เป็นตัว best)
# ดึงตารางผลลัพธ์จาก compare_models
compare_grid = pull().sort_values(by='RMSE') # ดึงและเรียงตาม RMSE
top_models_names = compare_grid.index[:3].tolist() # เอาชื่อ 3 อันดับแรก

# เพิ่มโมเดลอื่นถ้ายังไม่มีใน list และไม่ใช่ตัวเดียวกับ base model
if base_model_for_comparison:
    base_model_name = type(base_model_for_comparison).__name__
    for model_name in top_models_names:
         # แปลงชื่อย่อเป็นชื่อ PyCaret ID (เช่น 'Light Gradient Boosting Machine' -> 'lightgbm')
         try:
             model_id = compare_grid.loc[model_name].name
             if model_id != compare_grid.loc[base_model_name].name and len(models_to_blend) < 3: # จำกัดแค่ 3 ตัวพอ
                 print(f"Adding {model_name} ({model_id}) to blend list...")
                 try:
                      models_to_blend.append(create_model(model_id, verbose=False))
                 except Exception as e:
                      print(f"Could not create model {model_id}: {e}")
         except KeyError:
              print(f"Could not find model ID for {model_name}")


if len(models_to_blend) > 1:
    print(f"\nAttempting to blend {len(models_to_blend)} models: {[type(m).__name__ for m in models_to_blend]}...")
    try:
        blender_model = blend_models(estimator_list=models_to_blend, optimize='RMSE')
        print("\nBlended Model Results (Cross-Validation Metrics):")
        print(blender_model)
    except NameError:
        print("Error: 'blend_models' not defined.")
    except Exception as e:
        print(f"Could not blend models: {e}")
else:
    print("\nNot enough diverse models to perform blending.")


# --- 6. Final Model Selection and Evaluation on Test Set ---
print("\n--- 6. Final Model Selection and Evaluation ---")

final_model = None
final_model_name = "N/A"
final_rmse_cv = float('inf') # RMSE จาก CV
final_mae_cv = float('inf') # MAE จาก CV

# เลือก Blender ถ้าดีกว่าตัวที่ Tune แล้ว (วัดจาก CV ตอน Blend/Tune)
if blender_model is not None:
     try:
         rmse_blender_cv = get_metrics(blender_model)['RMSE']
         if base_model_for_comparison:
              rmse_tuned_cv = get_metrics(base_model_for_comparison)['RMSE']
              if rmse_blender_cv < rmse_tuned_cv:
                  final_model = blender_model
                  final_model_name = "Blender"
                  final_rmse_cv = rmse_blender_cv
                  final_mae_cv = get_metrics(blender_model)['MAE']
                  print(f"Selected Blender model (CV RMSE: {final_rmse_cv:.4f})")
              else:
                  final_model = base_model_for_comparison
                  final_model_name = f"Tuned_{type(final_model).__name__}"
                  final_rmse_cv = rmse_tuned_cv
                  final_mae_cv = get_metrics(final_model)['MAE']
                  print(f"Selected Tuned Single model (CV RMSE: {final_rmse_cv:.4f}) as it was better/equal to Blender.")
         else: # กรณีไม่มี base model (ไม่ควรเกิด)
              final_model = blender_model
              final_model_name = "Blender"
              final_rmse_cv = rmse_blender_cv
              final_mae_cv = get_metrics(blender_model)['MAE']
              print(f"Selected Blender model (CV RMSE: {final_rmse_cv:.4f})")

     except Exception as e:
          print(f"Error comparing blender and tuned model: {e}. Falling back to tuned/best model.")
          if base_model_for_comparison:
               final_model = base_model_for_comparison
               final_model_name = f"Tuned/Best_{type(final_model).__name__}"
               try:
                    final_rmse_cv = get_metrics(final_model)['RMSE']
                    final_mae_cv = get_metrics(final_model)['MAE']
               except: pass # ถ้า get_metrics ไม่ได้ ก็ใช้ค่า inf
          else:
               final_model = None # ไม่มีโมเดลให้เลือก

# ถ้าไม่มี Blender หรือ Blender ไม่ดีกว่า ก็ใช้ตัวที่ Tune แล้ว (หรือ Best เดิมถ้า Tune ไม่ได้)
elif base_model_for_comparison is not None:
    final_model = base_model_for_comparison
    final_model_name = f"Tuned/Best_{type(final_model).__name__}"
    try:
         final_rmse_cv = get_metrics(final_model)['RMSE']
         final_mae_cv = get_metrics(final_model)['MAE']
    except: pass
    print(f"Selected Tuned/Best Single model (CV RMSE: {final_rmse_cv:.4f})")
else:
    print("No suitable model could be selected.")


# ประเมิน Final Model บน Test Set
test_rmse = float('nan')
test_mae = float('nan')
if final_model is not None:
    print(f"\nPredicting on the Test set using the final selected model ({final_model_name})...")
    predictions_final_test = predict_model(final_model)
    print("\nSample Predictions on Test Set (Final Model):")
    print(predictions_final_test.head())

    # ดึง Metrics จากตารางผลลัพธ์ predict_model ล่าสุด
    final_score_grid = pull()
    try:
        test_rmse = final_score_grid['RMSE'][0]
        test_mae = final_score_grid['MAE'][0]
        print("\n--- Final Metrics on Test Set ---")
        print(f"Final Model Type: {final_model_name}")
        print(f"RMSE on Test Set: {test_rmse:.4f} meters")
        print(f"MAE on Test Set:  {test_mae:.4f} meters")
    except (KeyError, IndexError, TypeError) as e:
        print(f"Could not extract metrics from the final prediction output: {e}")
else:
    print("\nNo final model was selected, cannot evaluate on test set.")


# --- 7. (Optional) Finalize Model and Save ---
# Finalize คือการเทรนโมเดลที่ดีที่สุดด้วยข้อมูลทั้งหมด (Train + Test)
# เหมาะสำหรับเตรียมโมเดลไปใช้งานจริง แต่อย่าใช้ Metric จากขั้นตอนนี้มาวัดประสิทธิภาพ

print("\n--- 7. Finalizing the Model (Training on ALL data) ---")
if final_model is not None:
    try:
        finalized_model = finalize_model(final_model)
        print("\nFinalized Model Configuration:")
        print(finalized_model)

        # บันทึกโมเดลที่ Finalize แล้ว
        save_model(finalized_model, 'final_calibrated_depth_model')
        print("\nFinalized model saved as 'final_calibrated_depth_model.pkl'")

        # ทดลองโหลดโมเดลกลับมาใช้ (ตัวอย่าง)
        loaded_model = load_model('final_calibrated_depth_model')
        print("\nModel loaded successfully.")
        new_pred = predict_model(loaded_model, data=data_for_setup.iloc[:5]) # ลองทำนาย 5 แถวแรก
        print("\nSample predictions using loaded finalized model:")
        print(new_pred)

    except NameError:
         print("Error: 'finalize_model' or 'save_model' not defined.")
    except Exception as e:
         print(f"An error occurred during finalization or saving: {e}")
else:
     print("\nNo final model to finalize.")

# --- คำนวณ RMSE/MAE แยกตาม Location บน Test Set ---
# ใช้ DataFrame predictions_final_test ที่ได้จาก predict_model(final_model)

if 'predictions_final_test' in locals() and isinstance(predictions_final_test, pd.DataFrame):
    print("\n--- Calculating Metrics per Location on Test Set ---")

    # คำนวณ Squared Error (SE) และ Absolute Error (AE)
    predictions_final_test['Squared_Error'] = (predictions_final_test['prediction_label'] - predictions_final_test['IntendedDistance_m'])**2
    predictions_final_test['Absolute_Error'] = abs(predictions_final_test['prediction_label'] - predictions_final_test['IntendedDistance_m'])

    # Group by Location และคำนวณ Mean Squared Error (MSE) และ Mean Absolute Error (MAE)
    metrics_by_location = predictions_final_test.groupby('Location').agg(
        MSE=('Squared_Error', 'mean'),
        MAE=('Absolute_Error', 'mean'),
        Count=('Location', 'size') # นับจำนวนข้อมูลแต่ละกลุ่มด้วย
    )

    # คำนวณ RMSE จาก MSE
    metrics_by_location['RMSE'] = np.sqrt(metrics_by_location['MSE'])

    # จัดเรียงคอลัมน์ใหม่เพื่อให้อ่านง่าย
    metrics_by_location = metrics_by_location[['RMSE', 'MAE', 'MSE', 'Count']]

    print("\nMetrics calculated per Location:")
    print(metrics_by_location)

    # แสดงผลแยกแต่ละสถานที่
    for location, metrics in metrics_by_location.iterrows():
        print(f"\nLocation: {location}")
        print(f"  RMSE: {metrics['RMSE']:.4f} meters ({metrics['RMSE']*100:.2f} cm)")
        print(f"  MAE:  {metrics['MAE']:.4f} meters ({metrics['MAE']*100:.2f} cm)")
        print(f"  Count: {int(metrics['Count'])} data points")

else:
    print("\nDataFrame 'predictions_final_test' not found or is not a DataFrame. Please run predict_model first.")


# --- 8. (Optional) Analyze Error of the Final Model ---
if final_model is not None:
     print("\n--- 8. Analyzing Error of the Final Model on Test Set ---")
     try:
          evaluate_model(final_model) # ดู plot ต่างๆ ของตัวที่เลือกเป็น final (ก่อน finalize)

          # คำนวณ Error = prediction - actual (บน Test set)
          predictions_final_test['Error'] = predictions_final_test['prediction_label'] - predictions_final_test['IntendedDistance_m']

          # Plot Histogram ของ Error
          plt.figure(figsize=(10, 6))
          plt.hist(predictions_final_test['Error'], bins=50, alpha=0.7)
          plt.xlabel("Prediction Error (Prediction - Actual) (m)")
          plt.ylabel("Frequency")
          plt.title(f"Histogram of Prediction Errors on Test Set ({final_model_name})")
          plt.grid(True)
          plt.show()

          # Plot Error vs Intended Distance
          plt.figure(figsize=(10, 6))
          plt.scatter(predictions_final_test['IntendedDistance_m'], predictions_final_test['Error'], alpha=0.1)
          plt.axhline(0, color='red', linestyle='--')
          plt.xlabel("Intended Distance (m)")
          plt.ylabel("Prediction Error (m)")
          plt.title("Prediction Error vs Intended Distance on Test Set")
          plt.grid(True)
          plt.show()

          # ดูสถิติ Error แยกตาม Location (ถ้ามี)
          if 'Location' in predictions_final_test.columns:
              print("\nError Statistics by Location:")
              print(predictions_final_test.groupby('Location')['Error'].agg(['mean', 'std', 'min', 'max', 'count']))

     except Exception as e:
          print(f"An error occurred during error analysis: {e}")

print("\n--- Script Finished ---")