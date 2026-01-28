#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV Embedding Generator with MPNet
=================================

Generates vector representations (embeddings) from resume texts.
Handles the 512-token limit of transformer models:
long resumes are automatically split or truncated.

Output:
- Numpy file:     output/embeddings/cv_embeddings.npy
- CSV file:       output/embeddings/cv_embeddings.csv
- Logs statistics on length and processing
"""

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =====================================================
# Configuration
# =====================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data/raw/resume_data.csv"
OUT_DIR = BASE / "data/processed/embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 512  # Model limit

# Functions
def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """
    Combines relevant resume fields into a single text per candidate.
    """
    cols = [
        "career_objective", "skills", "educational_institution_name",
        "degree_names", "major_field_of_studies", "responsibilities",
        "certification_skills", "languages"
    ]
    existing = [c for c in cols if c in df.columns]
    combined = df[existing].fillna("").agg(" ".join, axis=1).str.strip()
    logger.info(f"Gebruikte kolommen: {existing}")
    return combined


def split_or_truncate(text: str, tokenizer, max_len=MAX_LENGTH):
    """
    Splits or truncates text if longer than the model limit.
    """
    tokens = tokenizer(text)["input_ids"]
    if len(tokens) <= max_len:
        return [text]
    else:
        chunks = []
        for i in range(0, len(tokens), max_len):
            sub_tokens = tokens[i:i + max_len]
            chunk_text = tokenizer.decode(sub_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks


def generate_embeddings(df: pd.DataFrame, model_name: str):
    """
    Generates embeddings for each resume (with length management).
    Returns numpy-array of (n_samples, 768).
    """
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformer(model_name)

    all_embeddings = []
    long_count = 0

    for text in tqdm(df["combined_text"], desc="Genereren van embeddings"):
        segments = split_or_truncate(text, tokenizer)
        if len(segments) > 1:
            long_count += 1
        seg_vectors = model.encode(segments, convert_to_numpy=True)
        avg_vector = np.mean(seg_vectors, axis=0)
        all_embeddings.append(avg_vector)

    logger.info(f"Total resumes processed: {len(all_embeddings)}")
    logger.info(f"Resumes longer than {MAX_LENGTH} tokens (split): {long_count}")
    return np.array(all_embeddings)


def save_embeddings(vectors: np.ndarray, df: pd.DataFrame, out_dir: str):
    """
    Saves embeddings in .npy and .csv format.
    """
    npy_path = os.path.join(out_dir, "cv_embeddings.npy")
    csv_path = os.path.join(out_dir, "cv_embeddings.csv")

    np.save(npy_path, vectors)
    logger.info(f"Numpy file saved: {npy_path}")

    emb_df = pd.DataFrame(vectors)
    emb_df.index = df.index
    emb_df.to_csv(csv_path, index=False)
    logger.info(f"CSV file saved: {csv_path}")

# Main runner
def main():
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["combined_text"] = combine_text_fields(df)

    logger.info("Starting embedding generation...")
    embeddings = generate_embeddings(df, MODEL_NAME)

    logger.info("Saving results...")
    save_embeddings(embeddings, df, OUT_DIR)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info("Done! Embeddings are ready for matching and clustering.")

# Execute
if __name__ == "__main__":
    main()
