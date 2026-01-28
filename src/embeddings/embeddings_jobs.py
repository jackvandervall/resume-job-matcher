#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Job Description Embedding Generator with MPNet.

Generates vector representations (embeddings) of job description texts.
Optimized with batch processing for faster processing.

Output:
- Numpy file:   output/embeddings/job_embeddings_sbert.npy
- Pickle file:  output/embeddings/job_embeddings_sbert.pkl
- CSV file:     output/embeddings/job_embeddings.csv (optional)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Any
import gc
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobDescriptionVectorizer:
    """
    A class to tokenize and vectorize job descriptions using an SBERT model.
    """

    def __init__(self, model_name: str = 'all-mpnet-base-v2', batch_size: int = 32):
        """
        Initialize the vectorizer with a specific SBERT model.

        Args:
            model_name: Name of the sentence transformer model to use
            batch_size: Batch size for encoding (adjust based on GPU memory)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embeddings = None
        self.text_columns = ['Job Description', 'Benefits', 'skills', 'Responsibilities']

    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading SBERT model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for job descriptions.

        Args:
            text: Raw text to preprocess

        Returns:
            Truncated word count
        """
        # Truncate long texts (SBERT has token limits)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        word_count = len(text.split())
        if word_count > 350:
            text = ' '.join(text.split()[:350])
            # logger.warning(f"Truncated text from {word_count} to 350 words") # Verbose

        return text
    
    def combine_text_features(self, row: pd.Series) -> str:
        """
        Combine relevant text columns into a single text for vectorization.
        """
        texts = []
        
        columns_to_embed = [
            "Job Description",
            "Skills",
            "Responsibilities",
            "Benefits",
            "Experience",
            "Qualifications",
            "Salary Range",
            "location",
            "Work Type",
            "Job Title",
            "Role",
            "Company profile",
            "Job Category", # Useful fields from NYC jobs
            "Business Title"
        ]

        for col_name in columns_to_embed:
            # Check if column exists
            if col_name not in row.index:
                continue
                
            # Get the text (which already has its prefix from preprocessing)
            val = row.get(col_name, '')
            text = self.preprocess_text(val)
            
            if text:
                texts.append(text)

        # Join everything with a separator
        return " | ".join(texts)


    def create_embeddings(self, df: pd.DataFrame, text_column: str = None) -> np.ndarray:
        """
        Create SBERT embeddings for job descriptions.

        Args:
            df: DataFrame containing job descriptions
            text_column: Column name containing text (if None, combines multiple columns)

        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()

        logger.info("Preparing text data for vectorization...")

        if text_column and text_column in df.columns:
            texts = df[text_column].apply(self.preprocess_text).tolist()
        else:
            texts = df.apply(self.combine_text_features, axis=1).tolist()

        logger.info(f"Vectorizing {len(texts)} job descriptions...")
        logger.info(f"Using batch size: {self.batch_size}")

        embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for better similarity search
            )
            embeddings.append(batch_embeddings)

            # Clear memory periodically
            if (i // self.batch_size) % 100 == 0:
                gc.collect()

        self.embeddings = np.vstack(embeddings)

        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path: str, metadata: Dict[str, Any] = None):
        """
        Save embeddings and metadata to disk.

        Args:
            embeddings: Numpy array of embeddings
            output_path: Path to save the embeddings
            metadata: Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_data = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'num_samples': embeddings.shape[0],
            'timestamp': timestamp,
            'metadata': metadata or {}
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Embeddings saved to: {output_path}")

        # Also save as numpy array for easier loading (standard format)
        np_path = output_path.replace('.pkl', '.npy')
        np.save(np_path, embeddings)
        logger.info(f"Embeddings also saved as numpy array: {np_path}")


def main():
    """Main function to run the vectorization process."""
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data/processed/job_descriptions_cleaned.csv"
    OUTPUT_PATH = BASE_DIR / "data/processed/job_embeddings_sbert.pkl"
    
    # Model configuration
    MODEL_NAME = 'all-mpnet-base-v2' # Consistent with CV embeddings
    BATCH_SIZE = 32  # Safer default for standard hardware
    
    logger.info("=== Job Description Vectorization with SBERT ===")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    
    try:
        # Load the cleaned dataset
        logger.info(f"Loading dataset from: {DATA_PATH}")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Input file not found: {DATA_PATH}")
            
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Initialize vectorizer
        vectorizer = JobDescriptionVectorizer(
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE
        )
        
        # Create embeddings
        embeddings = vectorizer.create_embeddings(df)
        
        # Prepare metadata
        metadata = {
            'original_dataset_shape': df.shape,
            'text_columns_used': vectorizer.text_columns,
            'preprocessing_steps': [
                'Handling missing values',
                'Removing duplicate data',
                'Applying text normalization',
                'Text combination from multiple columns',
                'Basic cleaning and truncation',
                'L2 normalization of embeddings'
            ]
        }
        
        # Save embeddings
        vectorizer.save_embeddings(embeddings, str(OUTPUT_PATH), metadata)
        
        # Print summary statistics
        logger.info("=== Vectorization Summary ===")
        logger.info(f"Total job descriptions processed: {embeddings.shape[0]}")
        logger.info(f"Embedding dimensions: {embeddings.shape[1]}")
        logger.info(f"Model used: {MODEL_NAME}")
        
        logger.info("Vectorization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during vectorization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
