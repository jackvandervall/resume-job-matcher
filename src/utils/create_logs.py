
import pandas as pd
from datetime import datetime 
from pathlib import Path

# Load the dataset
print("Loading dataset...")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
data_path = BASE_DIR / "data/raw/job_descriptions.csv"
if not data_path.exists():
    print(f"Warning: Data file not found at {data_path}")
else:
    df = pd.read_csv(data_path)

# Create log file
log_dir = BASE_DIR / "docs/logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"eda_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Create log function
def log_and_print(message, file_path=log_file):
    """Function to both print and log messages"""
    print(message)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
        
log_and_print(f"Dataset shape: {df.shape}")
