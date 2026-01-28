import pandas as pd
import numpy as np
import logging
import nltk
from nltk.corpus import stopwords
import re
import html
import unicodedata

nltk.download('stopwords')

logger = logging.getLogger(__name__)

# =====================================================
# Apply all preprocessing steps subsequently
# =====================================================
def preprocess_job_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline for NYC Jobs dataset.
    
    Args:
        df: Raw job descriptions DataFrame
        
    Returns:
        Cleaned DataFrame ready for vectorization
    """
    logger.info(f"Starting preprocessing. Input shape: {df.shape}")
    
    # Standardize column names for downstream compatibility
    # Mapping NYC Jobs columns to our generic schema
    rename_map = {
        "Business Title": "Job Title",
        "Job Description": "Job Description",
        "Minimum Qual Requirements": "Qualifications",
        "Preferred Skills": "Skills",
        "Work Location": "Location",
        "Agency": "Company"
    }
    df = df.rename(columns=rename_map)
    
    # Select only relevant columns
    keep_columns = [
        "Job ID", "Job Title", "Job Description", 
        "Qualifications", "Skills", "Location", "Company",
        "Salary Range From", "Salary Range To", "Salary Frequency"
    ]
    
    # Filter for columns that actually exist (intersection)
    existing_cols = [c for c in keep_columns if c in df.columns]
    df = df[existing_cols]
    
    logger.info(f"After column selection/renaming: {df.shape}")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df)
    logger.info(f"After missing values: {df.shape}")
    
    # Step 2: Remove duplicates 
    df = remove_duplicates(df)
    logger.info(f"After duplicate removal: {df.shape}")
    
    # Step 3: Text normalization
    df = normalize_text_columns(df)
    logger.info(f"After text normalization: {df.shape}")

    # Step 4: Handle special characters
    df = handle_special_characters(df)
    logger.info(f"After handling special characters: {df.shape}")
        
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df

# =====================================================
# Missing Values
# =====================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    """
    df_clean = df.copy()
    
    # Critical text columns
    text_columns = ["Job Title", "Job Description", "Qualifications", "Skills"]
    existing_text_cols = [c for c in text_columns if c in df_clean.columns]

    # Replace NaN with "missing_text"
    df_clean[existing_text_cols] = df_clean[existing_text_cols].fillna("missing_text")
    
    return df_clean

# =====================================================
# Duplicate Removal
# =====================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate job postings based on Job ID or content content.
    """
    df_clean = df.copy()
    # Drop duplicates based on Job ID if it exists
    if "Job ID" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["Job ID"])
    else:
        df_clean = df_clean.drop_duplicates()
        
    return df_clean

# =====================================================
# Text Normalization
# =====================================================
def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean text columns.
    Adds prefixes to give context to the model.
    """
    df_clean = df.copy()
    
    # Helper to prefix text
    def text_voorzetten(df_c, column, prefix):
        if column in df_c.columns:
            df_c[column] = prefix + df_c[column].astype(str)
        return df_c
    
    df_clean = text_voorzetten(df_clean, "Job Title", "Job Title: ")
    df_clean = text_voorzetten(df_clean, "Location", "Location: ")
    df_clean = text_voorzetten(df_clean, "Job Description", "Job Description: ")
    df_clean = text_voorzetten(df_clean, "Qualifications", "Qualifications: ")
    df_clean = text_voorzetten(df_clean, "Skills", "Skills: ")
    df_clean = text_voorzetten(df_clean, "Company", "Company: ")
    
    # Create a combined Salary string if possible
    if "Salary Range From" in df_clean.columns and "Salary Range To" in df_clean.columns:
        df_clean["Salary"] = "Salary: " + df_clean["Salary Range From"].astype(str) + " - " + df_clean["Salary Range To"].astype(str)
        if "Salary Frequency" in df_clean.columns:
            df_clean["Salary"] = df_clean["Salary"] + " (" + df_clean["Salary Frequency"].astype(str) + ")"
    
    # Valid columns to stopwords clean
    target_cols = ["Job Description", "Qualifications", "Skills"]
    
    # Ensure the stopwords resource is downloaded
    nltk_stopwords = set(stopwords.words("english"))

    def clean_text(text):
        if pd.isna(text):
            return text       
        # Simple extraction of words
        words = str(text).split()
        filtered = [w for w in words if w.lower() not in nltk_stopwords]
        return " ".join(filtered)

    for col in target_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_text)

    return df_clean

# =====================================================
# Special Characters
# =====================================================
def handle_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle special characters in the data.
    """
    df_clean = df.copy()

    # Helpers (compiled regex for speed)
    HTML_TAG_RX   = re.compile(r"<[^>]+>")
    CTRL_CHARS_RX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    MULTISPACE_RX = re.compile(r"\s+")
    # Allow letters/digits + common punctuation
    ALLOWED_CHARS_RX = re.compile(r"[^a-z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\'\"\-\/\&\%\+\@\#\$\€\£]")

    # Small mojibake/oddities
    REPAIRS = [
        ("â€™","’"), ("â€˜","‘"), ("â€œ","“"), ("â€\x9d","”"), ("â€\x9c","“"),
        ("â€“","–"), ("â€”","—"), ("â€¢","•"), ("â€¦","…"),
        ("Ã¡","á"), ("Ãà","à"), ("Ãâ","â"), ("Ãä","ä"),
        ("Ãã","ã"), ("Ãå","å"), ("Ãç","ç"),
        ("Ãé","é"), ("Ãè","è"), ("Ãê","ê"), ("Ãë","ë"),
        ("Ãí","í"), ("Ãì","ì"), ("Ãî","î"), ("Ãï","ï"),
        ("Ãñ","ñ"),
        ("Ãó","ó"), ("Ãò","ò"), ("Ãô","ô"), ("Ãõ","õ"), ("Ãö","ö"),
        ("Ãú","ú"), ("Ãù","ù"), ("Ãû","û"), ("Ãü","ü"),
        ("ÃŸ","ß"),
        ("Ã",""), ("Â",""),
    ]

    def _clean_cell(x: str | None) -> str | float | None:
        if pd.isna(x) or not isinstance(x, str):
            return x

        s = x
        s = html.unescape(s)
        s = HTML_TAG_RX.sub(" ", s)
        s = unicodedata.normalize("NFKC", s)
        s = CTRL_CHARS_RX.sub(" ", s)

        for a, b in REPAIRS:
            if a in s:
                s = s.replace(a, b)

        s = s.lower()
        s = ALLOWED_CHARS_RX.sub(" ", s)
        s = MULTISPACE_RX.sub(" ", s).strip()

        # Length guard
        if len(s) > 0 and len(s) < 5:
            return np.nan

        return s

    obj_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df_clean[c] = df_clean[c].apply(_clean_cell)

    return df_clean

# ====================================================
# INTEGRATION WITH VECTORIZER
# =====================================================
def run_complete_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.
    """
    # Load data
    logger.info(f"Loading data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    
    # Run preprocessing
    df_clean = preprocess_job_descriptions(df_raw)
    
    # Save cleaned data
    logger.info(f"Saving cleaned data to: {output_path}")
    df_clean.to_csv(output_path, index=False)
    
    # Print summary
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Original rows: {df_raw.shape[0]:,}")
    print(f"Final rows: {df_clean.shape[0]:,}")
    print(f"Rows removed: {df_raw.shape[0] - df_clean.shape[0]:,}")
    print(f"Percentage kept: {(df_clean.shape[0] / df_raw.shape[0]) * 100:.1f}%")
    
    return df_clean

# =====================================================
# USAGE EXAMPLE
# =====================================================
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # File paths
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data/raw/job_descriptions.csv"
    OUTPUT_FILE = BASE_DIR / "data/processed/job_descriptions_cleaned.csv"
    
    # Ensure processed directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    df_processed = run_complete_pipeline(str(INPUT_FILE), str(OUTPUT_FILE))