import pandas as pd
import os
from datetime import datetime
import io

# Load the dataset
file_path = "/home/admin-groep34/job-descriptions/data/raw/resume_data.csv"
print("Loading dataset...")

df = pd.read_csv(file_path)

# Drop unused columns before eda
drop_columns = [
    "passing_years", "educational_results", "result_types",
    "professional_company_names", "company_urls", "extra_curricular_organization_names",
    "extra_curricular_organization_links", "certification_providers", "", 
    "online_links", "issue_dates", "expiry_dates",
    # experiencere_requirement gaat waarschijnlijk over de duratie van de gevolgde opleiding.
    "experiencere_requirement", "age_requirement", "matched_score",
    ]

df = df.drop(columns=drop_columns, errors="ignore")

# Create log file
log_file = f"/home/admin-groep34/job-descriptions/notebooks/jack/logs/eda_logs/eda_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_and_print(message, file_path=log_file):
    """Function to both print and log messages"""
    print(message)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(str(message) + '\n')

# === START LOGGING ===
log_and_print(f"=== Exploratory Data Analysis ===")
log_and_print(f"Dataset: {file_path}")
log_and_print(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Log column drops
log_and_print(f"\n=== Column Drops ===")
log_and_print(f"Dropped columns: {drop_columns}")

# Shape
log_and_print(f"\n=== Rows and Columns ===")
log_and_print(f"Dataset shape: {df.shape}")
log_and_print(f"Number of rows: {df.shape[0]:,}")
log_and_print(f"Number of columns: {df.shape[1]}")

# Info
log_and_print(f"\n=== Info ===")
buf = io.StringIO()
df.info(buf=buf)
log_and_print(buf.getvalue())

# Head
log_and_print(f"\n=== Head ===")
log_and_print(df.head().to_string())

# Describe
log_and_print(f"\n=== Describe (Numerical Columns) ===")
log_and_print(df.describe().to_string())

# Nulls
log_and_print(f"\n=== Missing Values ===")
log_and_print(df.isnull().sum().to_string())

# Duplicates
log_and_print(f"\n=== Duplicates ===")
log_and_print(f"Number of duplicate rows: {df.duplicated().sum():,}")

# Column Data Types
log_and_print(f"\n=== Column Data Types ===")
log_and_print(df.dtypes.to_string())

# Unique Values per Column
log_and_print(f"\n=== Unique Values per Column ===")
log_and_print(df.nunique().to_string())

# Check for non-existing job offers
if 0 in df["# Of Positions"].unique():
    print("The column contains job descriptions with 0 positions.")
else:
    pass