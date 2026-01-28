import pandas as pd
import numpy as np
import logging
import re
import html
import unicodedata
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# =====================================================
# Apply all preprocessing steps subsequently
# =====================================================
def preprocess_resume_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline combining all preprocessing contributions.
    
    Args:
        df: Raw resume_data DataFrame
        
    Returns:
        Cleaned DataFrame ready for vectorization
    """
    logger.info(f"Starting preprocessing. Input shape: {df.shape}")
    
    # Column drops
    drop_columns = [
    "passing_years", "educational_results", "result_types",
    "professional_company_names", "company_urls", "extra_curricular_organization_names",
    "extra_curricular_organization_links", "certification_providers", 
    "online_links", "issue_dates", "expiry_dates",
    "experiencere_requirement", "age_requirement", "matched_score",
    ]
    
    df = df.drop(columns=drop_columns, errors='ignore')
    logger.info(f"After column drops: {df.shape}")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df)
    logger.info(f"After missing values: {df.shape}")
    
    # Step 2: Remove duplicates 
    df = remove_duplicates(df)
    logger.info(f"After duplicate removal: {df.shape}")
    
    # Step 3: Text normalization
    df = normalize_text_columns(df)
    logger.info(f"After text normalization: {df.shape}")
        
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df

# =====================================================
# Missing Values
# =====================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
        
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """

    df_clean = df.copy()
    # Columns that are important for classification and contain missing values
    text_columns = [
    "address", "career_objective", "skills",
    "educational_institution_name", "degree_names", "major_field_of_studies",
    "start_dates", "end_dates", "related_skils_in_job", 
    "positions", "locations", "extra_curricular_activity_types",
    "role_positions", "languages", "proficiency_levels",
    "certification_skills", "skills_required"
    ]

    # Replace NaN with "missing_text" in those columns
    # Missing values are replaced with a consistent textual indicator
    # to preserve semantic meaning during embedding.
    df[text_columns] = df[text_columns].fillna("missing_text")

    return df_clean

# =====================================================
# Duplicate Removal
# =====================================================  
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates.
        
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    df_clean = df.copy()
    
    df_clean = df.drop_duplicates()
    
    return df_clean

# =====================================================
# Text Normalization
# =====================================================
def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean text columns.
        
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized text
    """
    df_clean = df.copy()
    
    # Text columns to normalize
    text_columns = ['educationaL_requirements', 'skills_required']
    
    def text_voorzetten(df_clean, column, prefix):
        df_clean[column] = prefix + df_clean[column].astype(str)
        return df_clean

    df_clean.columns = [c.lstrip('\ufeff').strip() for c in df_clean.columns]

    df_clean = text_voorzetten(df_clean, "address", "address: ")
    df_clean = text_voorzetten(df_clean, "career_objective", "career objective: ")
    df_clean = text_voorzetten(df_clean, "skills", "skills: ")
    df_clean = text_voorzetten(df_clean, "educational_institution_name", "educational institution: ")
    df_clean = text_voorzetten(df_clean, "degree_names", "degree: ")
    df_clean = text_voorzetten(df_clean, "major_field_of_studies", "field of study: ")
    df_clean = text_voorzetten(df_clean, "start_dates", "start date: ")
    df_clean = text_voorzetten(df_clean, "end_dates", "end date: ")
    df_clean = text_voorzetten(df_clean, "related_skils_in_job", "related skills in job: ")
    df_clean = text_voorzetten(df_clean, "positions", "position: ")
    df_clean = text_voorzetten(df_clean, "locations", "location: ")
    df_clean = text_voorzetten(df_clean, "responsibilities", "responsibilities: ")
    df_clean = text_voorzetten(df_clean, "extra_curricular_activity_types", "extracurriculars: ")
    df_clean = text_voorzetten(df_clean, "role_positions", "role position: ")
    df_clean = text_voorzetten(df_clean, "languages", "languages: ")
    df_clean = text_voorzetten(df_clean, "proficiency_levels", "language proficiency: ")
    df_clean = text_voorzetten(df_clean, "certification_skills", "certification skills: ")
    df_clean = text_voorzetten(df_clean, "job_position_name", "position name: ")
    df_clean = text_voorzetten(df_clean, "educationaL_requirements", "educational requirement: ")
    df_clean = text_voorzetten(df_clean, "responsibilities.1", "responsibilities: ")
    df_clean = text_voorzetten(df_clean, "skills_required", "skills: ")

    df_clean = df_clean.applymap(lambda x: str(x).replace('\n', ', ') if isinstance(x, str) else x)

    # Ensure the stopwords resource is downloaded (run once in setup)
    # Remove standard English stopwords from text columns with nltk
    nltk_stopwords = set(stopwords.words("english"))

    # Function for removing standard stopwords
    def clean_text(text):
        if pd.isna(text):
            return text       
        words = str(text).split()
        filtered = [w for w in words if w.lower() not in nltk_stopwords]
        return " ".join(filtered)

    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_text)
        
    return df_clean

# =====================================================
# Special Characters
# =====================================================
def handle_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle special characters in the data.
        
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with special characters handled
    """
    df_clean = df.copy()

    # --- regex ---
    HTML_TAG_RX   = re.compile(r"<[^>]+>")
    CTRL_CHARS_RX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    MULTISPACE_RX = re.compile(r"\s+")
    # letters/digits + common punctuation
    ALLOWED_CHARS_RX = re.compile(r"[^a-z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\'\"\-\/\&\%\+\@\#\$\€\£]")
    # Standardize separators in skills fields (; | / • -> ,)
    SKILL_SEP_RX = re.compile(r"[;\|\/•]+")

    # Common mojibake/encoding repairs
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
        ("Ã",""), ("Â",""), (" ",""),
    ]

    def _clean_cell(x: str) -> str:
        s = html.unescape(x)                 # Decode HTML entities
        s = HTML_TAG_RX.sub(" ", s)          # Remove HTML tags
        s = unicodedata.normalize("NFKC", s) # Normalize Unicode
        s = CTRL_CHARS_RX.sub(" ", s)        # Remove control characters
        for a, b in REPAIRS:                 # Fix mojibake
            if a in s:
                s = s.replace(a, b)
        s = s.lower()                        # Lowercase for consistency
        s = ALLOWED_CHARS_RX.sub(" ", s)
        s = MULTISPACE_RX.sub(" ", s).strip()
        return s

    obj_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df_clean[c] = df_clean[c].apply(lambda v: _clean_cell(v) if isinstance(v, str) else v)
        # If this is a skills column, standardize separators
        if "skill" in c.lower():
            df_clean[c] = df_clean[c].apply(
                lambda s: MULTISPACE_RX.sub(" ", SKILL_SEP_RX.sub(",", s)).strip() if isinstance(s, str) else s
            )
            # Fix double commas and spaces
            df_clean[c] = df_clean[c].str.replace(r"\s*,\s*", ", ", regex=True)\
                                     .str.replace(r",\s*,+", ", ", regex=True)\
                                     .str.strip(", ")

    return df_clean

# =====================================================
# INTEGRATION WITH VECTORIZER
# =====================================================
def run_complete_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save cleaned CSV
        
    Returns:
        Cleaned DataFrame
    """
    # Load data
    logger.info(f"Loading data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    
    # Run preprocessing
    df_clean = preprocess_resume_data(df_raw)
    
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
    
    # File paths (CV data)
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data/raw/resume_data.csv"
    OUTPUT_FILE = BASE_DIR / "data/processed/resume_cleaned.csv"
    
    # Ensure processed directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    df_processed = run_complete_pipeline(str(INPUT_FILE), str(OUTPUT_FILE))