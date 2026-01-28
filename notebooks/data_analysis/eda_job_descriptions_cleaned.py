import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
file_path = "/home/admin-groep34/job-descriptions/data/processed/job_descriptions_cleaned.csv"
print("Loading dataset...")

df = pd.read_csv(file_path)

# Create log file
log_file = f"/home/admin-groep34/job-descriptions/notebooks/jack/logs/eda_logs/full_row_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_and_print(message, file_path=log_file):
    """Function to both print and log messages"""
    print(message)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(str(message) + '\n')

def count_words_in_text(text):
    """Count words in a text string, handling NaN values"""
    if pd.isna(text):
        return 0
    return len(str(text).split())

def get_row_word_count(row):
    """Count total words across all columns in a row"""
    total_words = 0
    for value in row:
        if pd.notna(value):
            # Only count text columns (skip numeric values)
            if isinstance(value, str) or pd.api.types.is_string_dtype(type(value)):
                total_words += count_words_in_text(value)
    return total_words

print("Identifying text columns...")
# Identify text columns (exclude numeric columns)
text_columns = []
for col in df.columns:
    if df[col].dtype == 'object':  # Usually text columns
        text_columns.append(col)

print(f"Text columns found: {text_columns}")

print("Calculating total word count per row across all text columns...")
# Calculate word count for entire rows (all text columns combined)
df['total_row_word_count'] = df[text_columns].apply(
    lambda row: sum(count_words_in_text(val) for val in row), 
    axis=1
)

# Also calculate per-column word counts for analysis
for col in text_columns:
    df[f'{col}_word_count'] = df[col].apply(count_words_in_text)

# Get top 10 longest rows by total word count
top_10_longest = df.nlargest(10, 'total_row_word_count')

# === START LOGGING ===
log_and_print(f"=== Full Row Word Count Analysis ===")
log_and_print(f"Dataset: {file_path}")
log_and_print(f"Text columns analyzed: {text_columns}")
log_and_print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"Total rows in dataset: {len(df):,}")

log_and_print(f"\n=== Total Row Word Count Statistics ===")
log_and_print(f"Mean total words per row: {df['total_row_word_count'].mean():.1f}")
log_and_print(f"Median total words per row: {df['total_row_word_count'].median():.1f}")
log_and_print(f"Max total words per row: {df['total_row_word_count'].max():,}")
log_and_print(f"Min total words per row: {df['total_row_word_count'].min():,}")
log_and_print(f"Standard deviation: {df['total_row_word_count'].std():.1f}")

# Distribution analysis
log_and_print(f"\n=== Word Count Distribution ===")
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(df['total_row_word_count'], p)
    log_and_print(f"{p}th percentile: {value:.0f} words")

print(f"\n✅ Complete row analysis finished! Check: {log_file}")