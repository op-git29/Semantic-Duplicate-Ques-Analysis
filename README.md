# Semantic Duplicate Question Detection using NLP

An end-to-end Natural Language Processing (NLP) project that automatically identifies whether two questions have the same semantic meaning (duplicate questions), built using SentenceTransformers (`all-MiniLM-L6-v2`) and TF-IDF baseline, and deployed as an interactive Streamlit web app.

## Overview

Online Q&A platforms like Quora and Stack Overflow receive thousands of questions every day. Many users ask different questions that have different wording but identical intent (e.g., *"How can I learn Python quickly?"* vs *"What is the fastest way to master Python?"*). Answering identical questions repeatedly fragments community knowledge and wastes expert contributor effort. This project builds a complete NLP pipeline that processes raw text, generates 384-dimensional dense semantic sentence embeddings, performs cosine similarity search, and serves real-time duplicate detection and answer recommendations through a clean web interface.

## Dataset

- **Source:** Quora Question Pairs Dataset (`quora_ques.csv` / `train.csv` — 404,351 question pairs)
- **Target variable:** `is_duplicate` (1 = Same meaning / Duplicate, 0 = Different meaning / Unique)
- **Class distribution:** ~63% Non-Duplicate (0), ~37% Duplicate (1) (imbalanced dataset — evaluated using F1-score)

**Columns used:** `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`.

## Project Workflow

1. **Data Exploration (EDA)** — verified data integrity (dropped 3 missing text rows out of 404k), analyzed class balance (63/37 split), and examined question length distributions (average ~11 words per question).
2. **Text Preprocessing** — cleaned text using lowercasing, regex URL/HTML stripping, contraction expansion (`don't` → `do not`), punctuation removal, tokenization, stopword removal, and NLTK lemmatization.
3. **Baseline Model (TF-IDF)** — built a traditional sparse TF-IDF vectorizer + Cosine Similarity baseline to demonstrate where keyword-matching algorithms fail on synonyms.
4. **Sentence Embeddings** — converted questions into 384-dimensional dense semantic vectors using pre-trained `all-MiniLM-L6-v2` SentenceTransformer.
5. **Similarity Search Engine** — computed vector cosine similarity scores between user queries and pre-computed database embeddings to retrieve top-5 most similar questions.
6. **Decision Threshold Tuning** — implemented similarity confidence thresholding ($\text{Similarity} \ge 60\% \rightarrow \text{Duplicate}$) to trigger existing answer reuse.
7. **Model Evaluation** — assessed performance on synonym failure cases and evaluated retrieval precision@K and F1-score.
8. **Deployment** — serialized models and embedded matrices into an interactive Streamlit web application.

## Results

| Metric / Model | TF-IDF Baseline | SentenceTransformer (`all-MiniLM-L6-v2`) |
|---|---|---|
| **Synonym Matching** (*"lose weight"* vs *"reduce fat"*) | 0.0% ❌ (Failed) | 60.3% ✅ (Passed) |
| **Punctuation Noise** (*Japanese* vs *English*) | 100.0% ❌ (Tricked) | 32.3% ✅ (Passed) |
| **Query Latency** | < 10ms | < 50ms |
| **F1-Score** | 0.64 | 0.82 |

**Key finding:** traditional TF-IDF suffered severely from vocabulary mismatch — failing completely on synonyms (giving 0% similarity to identical meanings) and getting tricked by shared filler words (giving 100% similarity to unrelated questions). SentenceTransformers solved both flaws by mapping questions into a dense semantic vector space where synonymous concepts point in identical directions.

## Tech Stack

- Python
- Pandas, NumPy
- NLTK (text preprocessing, stopword removal, lemmatization)
- Scikit-learn (TF-IDF vectorizer, evaluation metrics)
- SentenceTransformers / PyTorch (`all-MiniLM-L6-v2` embedding model)
- Streamlit (web application interface)

## Project Structure

```
Semantic-Duplicate-Question-Detection/
├── data/
│   └── quora_ques.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_TFIDF_Baseline.ipynb
│   └── 04_Sentence_Embeddings.ipynb
├── models/
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/op-git29/Semantic-Duplicate-Question-Detection.git
cd Semantic-Duplicate-Question-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Or explore the notebooks
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

## Future Improvements

- Scale vector similarity search using FAISS (Facebook AI Similarity Search) or a vector database (Qdrant / Milvus) for sub-millisecond search across millions of questions
- Fine-tune `all-MiniLM-L6-v2` directly on the Quora Question Pairs domain using MultipleNegativesRankingLoss
- Add voice-based question input in Streamlit using speech-to-text libraries
- Deploy to cloud hosting (Streamlit Community Cloud / HuggingFace Spaces)
