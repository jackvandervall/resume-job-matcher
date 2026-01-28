# Resume Job Matcher

## Description
This project implements a machine learning system to semantically match resumes with job descriptions using Sentence Transformers. It moves beyond keyword matching by calculating similarity scores based on the conceptual meaning of skills and experiences. The system includes an interactive interface for single-case comparisons, batch processing, and searching through a pre-indexed job database.

## Key Features
* Processes raw resume and job description data through customized cleaning pipelines.
* Generates vector embeddings for candidate profiles using the `all-mpnet-base-v2` model.
* Visualizes matching scores and detailed alignment metrics via an interactive dashboard.
* Automates batch production of suitability rankings for large candidate pools.
* Searches pre-indexed job embeddings to find matching positions for a given CV.

## Technologies Used
Python, Pandas, PyTorch, Sentence Transformers, Streamlit, Scikit-learn, NLTK.

## Installation
```bash
git clone https://github.com/jackvandervall/resume-job-matcher.git
uv sync
```

## Usage
```bash
# Run the preprocessing pipeline (one-time setup)
uv run python src/data_preprocessing/preprocess_jobs.py
uv run python src/data_preprocessing/preprocess_cv.py

# Generate embeddings (one-time setup)
uv run python src/embeddings/embeddings_jobs.py
uv run python src/embeddings/embeddings_cv.py

# Launch the web application
uv run streamlit run app.py
```

![Application Demo](https://github.com/user-attachments/assets/594befc1-e062-4636-8e61-ac84be07b8b9)

## Credits
Developed by Jack van der Vall in collaboration with Najah Khalifa, Celine Scova Righini and Brendan van der Sman.

## License
This project is licensed under the MIT License.
