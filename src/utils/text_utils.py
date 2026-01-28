"""
Text Preprocessing Utilities.

Reusable text cleaning functions for both batch preprocessing and
on-the-fly user input cleaning in the Streamlit UI.
"""

import re
import html
import unicodedata
from typing import Optional

# =====================================================
# REGEX PATTERNS (compiled once for performance)
# =====================================================
HTML_TAG_RX = re.compile(r"<[^>]+>")
CTRL_CHARS_RX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MULTISPACE_RX = re.compile(r"\s+")
ALLOWED_CHARS_RX = re.compile(r"[^a-z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\'\"\-\/\&\%\+\@\#\$]")

# Common mojibake/encoding repairs
REPAIRS = [
    ("a]!", "'"),
    ("a]~", "'"),
    ("a]o", '"'),
    ("a]", ""),
]


def preprocess_text(text: Optional[str], lowercase: bool = True) -> str:
    """Clean and normalize a single text string."""
    if text is None or not isinstance(text, str):
        return ""
    
    if not text.strip():
        return ""
    
    s = text
    
    # Decode HTML entities
    s = html.unescape(s)
    
    # Remove HTML tags
    s = HTML_TAG_RX.sub(" ", s)
    
    # Normalize Unicode
    s = unicodedata.normalize("NFKC", s)
    
    # Remove control characters
    s = CTRL_CHARS_RX.sub(" ", s)
    
    # Fix mojibake encoding issues
    for old, new in REPAIRS:
        if old in s:
            s = s.replace(old, new)
    
    # Lowercase for consistency
    if lowercase:
        s = s.lower()
    
    # Remove non-allowed characters
    s = ALLOWED_CHARS_RX.sub(" ", s)
    
    # Normalize whitespace
    s = MULTISPACE_RX.sub(" ", s).strip()
    
    return s


def truncate_text(text: str, max_words: int = 350) -> str:
    """Truncate text to a maximum number of words."""
    if not text:
        return ""
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    return " ".join(words[:max_words])


def clean_and_truncate(text: Optional[str], max_words: int = 350) -> str:
    """Convenience function: preprocess and truncate in one call."""
    cleaned = preprocess_text(text)
    return truncate_text(cleaned, max_words)
