from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "Dataset" / "car_data_original.csv"
OUTPUT_FILE = BASE_DIR / "Dataset" / "car_data_cleaned.csv"

data = pd.read_csv(INPUT_FILE)
data.columns = [column.strip().replace(" ", "_") for column in data.columns]

for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].astype(str).str.strip()

duplicate_count = data.duplicated().sum()
data = data.drop_duplicates().reset_index(drop=True)
data.to_csv(OUTPUT_FILE, index=False)

print(f"Rows before cleaning: {len(pd.read_csv(INPUT_FILE))}")
print(f"Duplicate rows removed: {duplicate_count}")
print(f"Rows after cleaning: {len(data)}")
print(f"Missing values after cleaning: {int(data.isna().sum().sum())}")
print(f"Cleaned dataset saved to: {OUTPUT_FILE}")
