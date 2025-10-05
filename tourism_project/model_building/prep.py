# for data manipulation
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# for Hugging Face dataset access & uploads
from huggingface_hub import HfApi, hf_hub_download

# Config (env-driven) get HF_TOKEN from env variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in environment. ")

api = HfApi(token=HF_TOKEN)

# Load dataset from HF

HF_DATASET_URL = f"hf://datasets/cbendale10/MLOps-Tourism-Prediction/tourism.csv"
df = pd.read_csv(HF_DATASET_URL)

print("Original shape:", df.shape)

# Remove unnamed/blank index columns

unnamed_cols = [c for c in df.columns if (str(c).strip() == "" or str(c).lower().startswith("unnamed"))]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True, errors="ignore")
    print("Dropped unnamed columns:", unnamed_cols)


# Minimal cleaning per schema

LABEL_COL = "ProdTaken"

ID_COLS = ["CustomerID"]

NUMERIC_COLS = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched"
]

# Drop unique identifiers if present
for c in ID_COLS:
    if c in df.columns:
        df.drop(columns=c, inplace=True, errors="ignore")

# Basic whitespace cleanup for object columns
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.strip()

# Normalize the Gender typo
if "Gender" in df.columns:
    df["Gender"].replace("Fe Male", "Female", inplace=True)

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped {before - len(df)} duplicate rows.")


print("\nLabel distribution (incl. NaN):")
print(df[LABEL_COL].value_counts(dropna=False))

# ---------------------------
# Split into X/y and train/test
# ---------------------------
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]


Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nShapes after split:")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape, "ytrain:", ytrain.shape, "ytest:", ytest.shape)


# Save locally (CSV)

out_dir = "tourism_project/data"

Xtrain_path = os.path.join(out_dir, "Xtrain.csv")
Xtest_path  = os.path.join(out_dir, "Xtest.csv")
ytrain_path = os.path.join(out_dir, "ytrain.csv")
ytest_path  = os.path.join(out_dir, "ytest.csv")

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("\nSaved CSVs:")
for p in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
    print(" -", p)

# ---------------------------
# Upload split files back to HF dataset repo
# (mirroring your sample: upload with just the filename at repo root)
# ---------------------------
files_to_upload = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # just the filename
        repo_id="cbendale10/MLOps-Tourism-Prediction",
        repo_type="dataset",
    )

print(f"\nUploaded splits to dataset repo: cbendale10/MLOps-Tourism-Prediction")
