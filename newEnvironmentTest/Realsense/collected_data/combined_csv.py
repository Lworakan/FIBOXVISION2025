import os
import pandas as pd
import re # สำหรับ Regular Expressions เพื่อดึงตัวเลขระยะทาง

# --- กำหนดค่าเริ่มต้น ---
# 1. เปลี่ยน path นี้ให้ชี้ไปยังโฟลเดอร์หลักที่มีโฟลเดอร์ย่อยของแต่ละสถานที่
#    ตัวอย่างเช่น โฟลเดอร์ 'collected_data' ที่มี 'outdoor1stfloor', 'thirdfloor' อยู่ข้างใน
root_data_dir = r'C:\Users\tawan\OneDrive\Documents\GitHub\FIBOXVISION2025\newEnvironmentTest\Realsense\collected_data'

# 2. กำหนดชื่อไฟล์ CSV ที่จะบันทึกข้อมูลที่รวมแล้ว
output_csv_path = r'C:\Users\tawan\OneDrive\Documents\GitHub\FIBOXVISION2025\newEnvironmentTest\Realsense\collected_data\combined_depth_data_for_automl.csv'
# --- สิ้นสุดการกำหนดค่า ---

all_dataframes = [] # ลิสต์สำหรับเก็บ DataFrame จากแต่ละไฟล์

print(f"Starting data aggregation from: {root_data_dir}")

# Regex pattern เพื่อค้นหาตัวเลขระยะทางในชื่อไฟล์ เช่น 'depth_values_4m.csv' -> ได้ '4'
# \d+ หมายถึง ตัวเลข 1 ตัวหรือมากกว่า
# อยู่ระหว่าง '_' และ 'm.csv'
distance_pattern = re.compile(r'_(\d+)m\.csv$')

# วนลูปผ่านทุกไฟล์และโฟลเดอร์ย่อยใน root_data_dir
for dirpath, dirnames, filenames in os.walk(root_data_dir):
    # dirpath คือ path ปัจจุบัน เช่น '...collected_data\outdoor1stfloor'
    # dirnames คือ ลิสต์ของโฟลเดอร์ย่อยใน dirpath
    # filenames คือ ลิสต์ของไฟล์ใน dirpath

    # ดึงชื่อสถานที่จาก path ปัจจุบัน
    # os.path.basename จะให้ชื่อส่วนสุดท้ายของ path (ชื่อโฟลเดอร์)
    location = os.path.basename(dirpath)

    # ถ้า path ปัจจุบันคือ root_data_dir เอง ให้ข้ามไป (เราสนใจโฟลเดอร์ย่อยที่เป็นสถานที่)
    if os.path.abspath(dirpath) == os.path.abspath(root_data_dir):
        print(f"Skipping root directory: {dirpath}")
        continue

    print(f"\nProcessing location: {location} in folder: {dirpath}")

    for filename in filenames:
        # ตรวจสอบว่าเป็นไฟล์ CSV ที่เราสนใจหรือไม่
        if filename.startswith('depth_values_') and filename.endswith('m.csv'):
            # ลองสกัดระยะทางจากชื่อไฟล์
            match = distance_pattern.search(filename)
            if match:
                # match.group(1) จะดึงเฉพาะส่วนที่อยู่ในวงเล็บ (\d+) ซึ่งคือตัวเลข
                intended_distance = int(match.group(1))
                file_path = os.path.join(dirpath, filename)
                print(f"  Reading file: {filename} (Distance: {intended_distance}m)")

                try:
                    # อ่านไฟล์ CSV
                    df_temp = pd.read_csv(file_path)

                    # --- เพิ่มคอลัมน์ใหม่ ---
                    # เพิ่มคอลัมน์ 'IntendedDistance_m'
                    df_temp['IntendedDistance_m'] = intended_distance
                    # เพิ่มคอลัมน์ 'Location'
                    df_temp['Location'] = location

                    # --- เลือกเฉพาะคอลัมน์ที่จำเป็น (ปรับได้ตามต้องการ) ---
                    # ในที่นี้เราเลือก 'IntendedDistance_m', 'Location', และ 'average_depth_m'
                    # ถ้าต้องการ timestamp ด้วย ก็เพิ่ม 'timestamp' เข้าไปในลิสต์
                    df_selected = df_temp[['IntendedDistance_m', 'Location', 'average_depth_m']]

                    # เปลี่ยนชื่อคอลัมน์ average_depth_m ให้สอดคล้องกัน (ถ้าจำเป็น)
                    # df_selected = df_selected.rename(columns={'average_depth_m': 'AverageDepth_m'})

                    # เพิ่ม DataFrame ที่ประมวลผลแล้วลงในลิสต์
                    all_dataframes.append(df_selected)

                except FileNotFoundError:
                    print(f"    Error: File not found at {file_path}")
                except pd.errors.EmptyDataError:
                     print(f"    Warning: File is empty {file_path}")
                except Exception as e:
                    print(f"    Error reading or processing file {file_path}: {e}")
            else:
                print(f"  Skipping file (could not extract distance): {filename}")
        else:
             # ถ้าไม่ต้องการเห็นไฟล์อื่นที่ถูกข้าม ก็คอมเมนต์บรรทัดนี้ออก
             # print(f"  Skipping non-matching file: {filename}")
             pass


# ตรวจสอบว่ามีข้อมูลที่อ่านได้หรือไม่
if not all_dataframes:
    print("\nNo data found or processed. Please check the 'root_data_dir' path and file structure.")
else:
    # รวม DataFrame ทั้งหมดในลิสต์ให้เป็น DataFrame เดียว
    print("\nConcatenating all data...")
    df_combined = pd.concat(all_dataframes, ignore_index=True)

    # แสดงข้อมูลเบื้องต้นของ DataFrame ที่รวมแล้ว
    print("\nCombined DataFrame Info:")
    df_combined.info()
    print("\nFirst 5 rows of combined data:")
    print(df_combined.head())
    print("\nLast 5 rows of combined data:")
    print(df_combined.tail())

    # --- บันทึกเป็นไฟล์ CSV ---
    try:
        df_combined.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # ใช้ utf-8-sig เผื่อมีภาษาไทย
        print(f"\nSuccessfully saved combined data to: {output_csv_path}")
        print(f"Total rows combined: {len(df_combined)}")
    except Exception as e:
        print(f"\nError saving combined data to {output_csv_path}: {e}")