"""
Resume-Job Matching Streamlit Application

A modern UI for matching resumes to job descriptions using semantic similarity.
Supports:
- Single CV vs Job comparison (ad-hoc)
- Batch CV analysis against a single job
- Search Jobs: query pre-indexed job embeddings with a CV
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

# Add src to path for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from utils.text_utils import preprocess_text, clean_and_truncate

# ============================================================================
# CONFIGURATION
# ============================================================================
# Use the same model as the embedding pipeline for consistency
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# Paths to pre-built embeddings
DATA_DIR = BASE_DIR / "data" / "processed"
JOB_EMBEDDINGS_PATH = DATA_DIR / "job_embeddings_sbert.npy"
JOB_METADATA_PATH = DATA_DIR / "job_descriptions_cleaned.csv"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CACHING & MODEL LOADING
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the sentence transformer model."""
    with st.spinner("🔄 Loading AI model... This may take a moment on first run."):
        model = SentenceTransformer(MODEL_NAME)
    return model


@st.cache_data(show_spinner=False)
def load_job_embeddings():
    """Load pre-built job embeddings from disk."""
    if not JOB_EMBEDDINGS_PATH.exists():
        return None, None
    
    embeddings = np.load(JOB_EMBEDDINGS_PATH)
    
    if JOB_METADATA_PATH.exists():
        metadata = pd.read_csv(JOB_METADATA_PATH)
    else:
        metadata = None
    
    return embeddings, metadata


def compute_embedding(model, text: str):
    """Compute embedding for a single text with preprocessing."""
    if not text or not text.strip():
        return None
    
    # Apply preprocessing for consistency with pre-indexed data
    cleaned_text = clean_and_truncate(text, max_words=350)
    
    if not cleaned_text:
        return None
    
    return model.encode(cleaned_text, convert_to_numpy=True, normalize_embeddings=True)


def calculate_match_score(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    sim = cosine_similarity(
        embedding1.reshape(1, -1), 
        embedding2.reshape(1, -1)
    )[0][0]
    
    return float(sim)


def get_score_label(score):
    """Get label based on score."""
    if score >= 0.7:
        return "🎯 Strong Match"
    elif score >= 0.4:
        return "⚡ Moderate Match"
    else:
        return "📊 Weak Match"


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.title("📄 Resume Matcher")
    st.markdown("*AI-powered resume-job matching using semantic similarity*")
    st.divider()
    
    # Load model
    model = load_model()
    
    # Load pre-built embeddings
    job_embeddings, job_metadata = load_job_embeddings()
    has_prebuilt = job_embeddings is not None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        modes = ["Single Match", "Batch Analysis"]
        if has_prebuilt:
            modes.append("Search Jobs")
        
        match_mode = st.radio(
            "Matching Mode",
            modes,
            help="Single Match: Compare one resume to one job. Batch: Compare multiple. Search Jobs: Find matching jobs from database."
        )
        
        st.divider()
        
        st.subheader("📊 Model Info")
        st.info(f"Using **{MODEL_NAME}** for semantic embeddings")
        
        if has_prebuilt:
            st.success(f"✅ {len(job_embeddings):,} pre-indexed jobs available")
        else:
            st.warning("⚠️ No pre-built job embeddings found")
        
        st.divider()
        
        st.subheader("🎯 Score Guide")
        st.markdown("""
        - 🟢 **70%+**: Strong match
        - 🟡 **40-70%**: Moderate match  
        - 🔴 **<40%**: Weak match
        """)
    
    # Main content based on mode
    if match_mode == "Single Match":
        render_single_match(model)
    elif match_mode == "Batch Analysis":
        render_batch_analysis(model)
    elif match_mode == "Search Jobs":
        render_search_jobs(model, job_embeddings, job_metadata)


def render_single_match(model):
    """Render single match comparison interface."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Job Description")
        job_text = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="Enter the job description including required skills, responsibilities, and qualifications...",
            key="job_input"
        )
    
    with col2:
        st.subheader("👤 Resume / CV")
        resume_text = st.text_area(
            "Paste the resume content here",
            height=300,
            placeholder="Enter the resume content including skills, experience, and education...",
            key="resume_input"
        )
    
    # Match button
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        if not job_text.strip() or not resume_text.strip():
            st.warning("⚠️ Please enter both a job description and a resume to analyze.")
            return
        
        with st.spinner("🧠 Analyzing semantic similarity..."):
            # Compute embeddings (preprocessing applied inside)
            job_embedding = compute_embedding(model, job_text)
            resume_embedding = compute_embedding(model, resume_text)
            
            # Calculate score
            score = calculate_match_score(job_embedding, resume_embedding)
        
        # Display results
        st.divider()
        st.header("📊 Match Results")
        
        # Score display
        col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
        with col_score2:
            score_label = get_score_label(score)
            
            if score >= 0.7:
                st.success(f"## {score:.1%}\n### {score_label}")
            elif score >= 0.4:
                st.warning(f"## {score:.1%}\n### {score_label}")
            else:
                st.error(f"## {score:.1%}\n### {score_label}")
        
        # Detailed metrics
        st.subheader("📈 Analysis Details")
        
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.metric("Job Description Words", len(job_text.split()))
        
        with metric_cols[1]:
            st.metric("Resume Words", len(resume_text.split()))
        
        with metric_cols[2]:
            st.metric("Embedding Dimensions", EMBEDDING_DIM)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if score >= 0.7:
            st.success("""
            **Excellent match!** This resume shows strong alignment with the job requirements.
            
            ✅ The candidate's skills and experience closely match the job description.  
            ✅ Consider prioritizing this candidate for interview.
            """)
        elif score >= 0.4:
            st.warning("""
            **Moderate match.** There's some overlap but room for improvement.
            
            ⚡ Some skills align, but there may be gaps in experience or qualifications.  
            ⚡ Consider reviewing specific requirements against the resume.
            """)
        else:
            st.error("""
            **Weak match.** Limited alignment between the resume and job description.
            
            📊 The candidate may lack key skills or experience required for this role.  
            📊 Consider other candidates or discuss transferable skills.
            """)


def render_batch_analysis(model):
    """Render batch analysis interface."""
    
    st.subheader("📁 Batch Analysis")
    st.info("Compare multiple resumes against a single job description.")
    
    # Job description input
    st.markdown("#### 📝 Job Description")
    job_text = st.text_area(
        "Enter the job description",
        height=200,
        placeholder="Enter the job description to match against...",
        key="batch_job"
    )
    
    # Resume upload
    st.markdown("#### 📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume files (TXT format)",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload multiple .txt files containing resume content"
    )
    
    if st.button("🔍 Analyze All Resumes", type="primary", use_container_width=True):
        if not job_text.strip():
            st.warning("⚠️ Please enter a job description.")
            return
        
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one resume file.")
            return
        
        # Process all resumes
        job_embedding = compute_embedding(model, job_text)
        
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            resume_text = file.read().decode('utf-8')
            resume_embedding = compute_embedding(model, resume_text)
            score = calculate_match_score(job_embedding, resume_embedding)
            
            results.append({
                'Filename': file.name,
                'Match Score': score,
                'Word Count': len(resume_text.split()),
                'Status': get_score_label(score)
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Display results
        st.divider()
        st.header("📊 Batch Results")
        
        # Sort by score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Match Score', ascending=False)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", len(results))
        
        with col2:
            st.metric("Avg. Score", f"{results_df['Match Score'].mean():.1%}")
        
        with col3:
            strong_matches = (results_df['Match Score'] >= 0.7).sum()
            st.metric("Strong Matches", strong_matches)
        
        with col4:
            best_score = results_df['Match Score'].max()
            st.metric("Best Match", f"{best_score:.1%}")
        
        # Results table
        st.subheader("📋 Ranked Results")
        
        # Format the dataframe for display
        display_df = results_df.copy()
        display_df['Match Score'] = display_df['Match Score'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv_data = results_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name="match_results.csv",
            mime="text/csv"
        )


def render_search_jobs(model, job_embeddings, job_metadata):
    """Render job search interface using pre-built embeddings."""
    
    st.subheader("🔍 Search Jobs")
    st.info("Find matching jobs from the pre-indexed database using your CV.")
    
    # CV input
    st.markdown("#### 👤 Your Resume / CV")
    cv_text = st.text_area(
        "Paste your resume content here",
        height=300,
        placeholder="Enter your skills, experience, education, and career objectives...",
        key="search_cv"
    )
    
    # Search settings
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of results", min_value=5, max_value=50, value=10)
    with col2:
        min_score = st.slider("Minimum match score", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    
    if st.button("🔍 Search Jobs", type="primary", use_container_width=True):
        if not cv_text.strip():
            st.warning("⚠️ Please enter your resume content.")
            return
        
        with st.spinner("🔍 Searching through job database..."):
            # Compute CV embedding
            cv_embedding = compute_embedding(model, cv_text)
            
            if cv_embedding is None:
                st.error("Failed to process CV text.")
                return
            
            # Calculate similarities with all jobs
            similarities = cosine_similarity(
                cv_embedding.reshape(1, -1),
                job_embeddings
            )[0]
            
            # Get top N results
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
        # Display results
        st.divider()
        st.header("📊 Matching Jobs")
        
        # Summary
        above_threshold = (similarities >= min_score).sum()
        st.metric("Jobs above threshold", above_threshold)
        
        # Results
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < min_score:
                continue
            
            job_info = {
                'Rank': len(results) + 1,
                'Match Score': f"{score:.1%}",
                'Score Raw': score
            }
            
            # Add metadata if available
            if job_metadata is not None and idx < len(job_metadata):
                row = job_metadata.iloc[idx]
                
                # Try common column names
                for col in ['Job Title', 'Business Title', 'Title', 'job_title']:
                    if col in row.index and pd.notna(row[col]):
                        job_info['Job Title'] = str(row[col])[:100]
                        break
                
                for col in ['Company', 'Agency', 'company']:
                    if col in row.index and pd.notna(row[col]):
                        job_info['Company'] = str(row[col])[:50]
                        break
                
                for col in ['Location', 'Work Location', 'location']:
                    if col in row.index and pd.notna(row[col]):
                        job_info['Location'] = str(row[col])[:50]
                        break
            
            results.append(job_info)
        
        if not results:
            st.warning("No jobs found above the minimum score threshold.")
            return
        
        results_df = pd.DataFrame(results)
        
        # Display table
        display_cols = [c for c in results_df.columns if c != 'Score Raw']
        st.dataframe(
            results_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Show detailed view for top result
        if results:
            st.subheader("🏆 Top Match Details")
            top_idx = top_indices[0]
            
            if job_metadata is not None and top_idx < len(job_metadata):
                row = job_metadata.iloc[top_idx]
                
                with st.expander("View full job details", expanded=True):
                    for col in row.index:
                        val = row[col]
                        if pd.notna(val) and str(val).strip():
                            st.markdown(f"**{col}:** {val}")


if __name__ == "__main__":
    main()
