import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Page configuration
st.set_page_config(
    page_title="Semantic Question Search",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for modern, clean UI
st.markdown("""
    <style>
    .header-box {
        text-align: center;
        padding: 20px 0px 10px 0px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 25px;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .match-number {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
    }
    .sim-badge {
        font-size: 0.9rem;
        font-weight: 700;
        color: #2563EB;
        background-color: #EFF6FF;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid #BFDBFE;
    }
    .dup-status {
        font-size: 0.85rem;
        font-weight: 700;
        color: #166534;
        background-color: #DCFCE7;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .new-status {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
        background-color: #F1F5F9;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .question-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1E293B;
        margin: 10px 0px 6px 0px;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown("""
<div class="header-box">
    <div class="main-title">🔍 Semantic Duplicate Question Search</div>
    <div class="sub-title">Find existing questions with identical meaning using Sentence Transformers</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Load Model & Dataset
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_dataset():
    data_path = 'data/quora_ques.csv'
    if not os.path.exists(data_path):
        data_path = 'quora_ques.csv'
    df = pd.read_csv(data_path).dropna(subset=['question1', 'question2'])
    
    # Store unique database questions
    questions = list(set(df['question1'].head(2000).tolist() + df['question2'].head(2000).tolist()))
    
    # Curated answers dictionary for smart recommendation
    answers_map = {
        "python": "To learn Python quickly: Master core syntax (variables, loops, functions), solve daily practice problems on platforms like LeetCode, and build hands-on projects (web scrapers, data analysis apps).",
        "weight": "To lose weight fast: Maintain a consistent caloric deficit, prioritize high-protein nutrition, perform resistance training combined with cardio, and get 7-8 hours of sleep.",
        "money": "Popular online income paths: Freelance programming/writing, developing digital tools/SaaS, affiliate marketing, or providing expert consulting.",
        "interview": "To clear tech interviews: Master Data Structures & Algorithms (focus on arrays, trees, graphs, dynamic programming), practice LeetCode 75, and conduct mock interviews."
    }
    
    return questions, answers_map

with st.spinner("Initializing Semantic Search Engine..."):
    model = load_model()
    questions_list, answers_map = load_dataset()

@st.cache_data
def get_embeddings():
    return model.encode(questions_list, convert_to_tensor=True)

embeddings_matrix = get_embeddings()

# -------------------------------------------------------------
# Single Clean Search Input
# -------------------------------------------------------------
user_query = st.text_input(
    "Ask your question:",
    placeholder="e.g., What is the fastest way to master Python?",
    key="search_input"
)

# Quick Suggestion Buttons
st.markdown("**Try sample questions:**")
col1, col2, col3 = st.columns(3)
if col1.button("💡 How to learn Python?"):
    user_query = "What is the best way to master Python?"
if col2.button("💡 How to lose weight?"):
    user_query = "What is the best method to reduce body fat?"
if col3.button("💡 How to earn money online?"):
    user_query = "What are good ways to make money from home?"

# -------------------------------------------------------------
# Search Logic & Output Display
# -------------------------------------------------------------
if user_query:
    st.markdown("---")
    
    # Compute similarity vector
    query_vector = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_vector, embeddings_matrix)[0]
    
    # Retrieve Top 5 highest scores
    top_5 = torch.topk(cosine_scores, k=5)
    
    best_score = top_5[0][0].item() * 100
    
    if best_score >= 60.0:
        st.success(f"🟢 **Duplicate Found!** (Highest Similarity: **{best_score:.1f}%**). Reusing existing answer below.")
    else:
        st.info(f"⚪ **New Question.** (Highest Similarity: **{best_score:.1f}%**). No direct duplicate found above 60%.")
        
    st.subheader("Top 5 Semantically Similar Questions:")
    
    # Loop over Top 5 matches and display cleanly
    for i, (score, idx) in enumerate(zip(top_5[0], top_5[1]), 1):
        matched_question = questions_list[idx]
        sim_percentage = score.item() * 100
        is_dup = sim_percentage >= 60.0
        
        status_tag = '<span class="dup-status">🟢 DUPLICATE</span>' if is_dup else '<span class="new-status">⚪ LOW MATCH</span>'
        
        # Get matching answer
        answer_text = "Standard answer: Focus on core fundamentals, practice consistently, and seek community guidance."
        for key, ans in answers_map.items():
            if key in user_query.lower() or key in matched_question.lower():
                answer_text = ans
                break
        
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="match-number">Match #{i}</span>
                <div>
                    <span class="sim-badge">Similarity: {sim_percentage:.1f}%</span>
                    {status_tag}
                </div>
            </div>
            <div class="question-text">"{matched_question}"</div>
            <div class="answer-box">
                <b>💡 Answer:</b> {answer_text}
            </div>
        </div>
        """, unsafe_allow_html=True)