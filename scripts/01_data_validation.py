# scripts/01_data_preparation.py

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import os # <-- ตรวจสอบว่ามีการ import os แล้ว

# --- ตั้งค่า Path (แก้ไขส่วนนี้) ---
# ใช้ os.path.join เพื่อสร้าง Path ที่ถูกต้องและเข้ากันได้กับทุก OS
BASE_DIR = os.getcwd() # หรือกำหนด Path ไปยังโฟลเดอร์โปรเจกต์ของคุณ
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "pulsar_data_train.csv") 
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
BALANCED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_balanced.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- 1. โหลดข้อมูล ---
print(f"Loading data from {RAW_DATA_PATH}...")
df = pd.read_csv(RAW_DATA_PATH)

# --- 2. ทำความสะอาดชื่อคอลัมน์ ---
print("Cleaning column names...")
df.columns = [col.strip().replace(' ', '_') for col in df.columns]
print("Column names cleaned.")

# --- 3. แยก Features (X) และ Target (y) ---
# แยกข้อมูลก่อนเพื่อไม่ให้ imputer ไปยุ่งกับคอลัมน์ target
X = df.drop("target_class", axis=1)
y = df["target_class"]

# --- 4. จัดการข้อมูลสูญหาย (Imputation) --- ## <<< ส่วนที่เพิ่มเข้ามา ##
print("\nHandling missing values (NaN)...")
if X.isnull().sum().sum() > 0:
    print(f"Found {X.isnull().sum().sum()} missing values. Applying SimpleImputer with mean strategy.")
    # สร้าง Imputer object เพื่อเติมค่า NaN ด้วยค่าเฉลี่ยของแต่ละคอลัมน์
    imputer = SimpleImputer(strategy='mean')
    
    # ทำการ fit_transform กับข้อมูล X
    X_imputed = imputer.fit_transform(X)
    
    # แปลงกลับเป็น DataFrame เพื่อให้ชื่อคอลัมน์ยังคงอยู่
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("Missing values handled.")
else:
    print("No missing values found.")

# --- 5. แก้ไขข้อมูลไม่สมดุลด้วย SMOTE ---
print("\nHandling class imbalance...")
print("Class distribution before SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y) # <-- ใช้ X ที่ผ่านการ Impute แล้ว

print("\nClass distribution after SMOTE:")
print(y_resampled.value_counts())

# --- 6. บันทึกข้อมูลที่เตรียมพร้อมแล้ว ---
print(f"\nSaving balanced data to {BALANCED_TRAIN_PATH}...")
balanced_df = pd.concat([X_resampled, y_resampled], axis=1)
balanced_df.to_csv(BALANCED_TRAIN_PATH, index=False)

print("\nData preparation complete! ✨")