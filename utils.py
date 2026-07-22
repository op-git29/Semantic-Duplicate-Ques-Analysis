# utils.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure downloads
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
}

def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text, remove_stop_words=True, use_lemmatization=True):
    """Clean and preprocess input text string for NLP models."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    words = text.split()
    if remove_stop_words:
        words = [w for w in words if w not in stop_words]
    if use_lemmatization:
        words = [lemmatizer.lemmatize(w) for w in words]
        
    return " ".join(words)