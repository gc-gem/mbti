import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import contractions
import re

# Download stopwords if they aren't already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Load models
tfidf_vect_ngram = joblib.load('tfidf_vect_ngram.pkl')
xgb_ie_model = joblib.load('xgb_ie_model.pkl')
xgb_ns_model = joblib.load('xgb_ns_model.pkl')
logreg_tf_model = joblib.load('logreg_tf_model.pkl')
logreg_jp_model = joblib.load('logreg_jp_model.pkl')

# Function to preprocess text input
def preprocess_text(text):
    # Extend contractions
    text = contractions.fix(text)

    # Substitute repeated characters (3 or more)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Remove pipes
    text = re.sub(r'\|+', ' ', text)

    # Remove special characters, digits, and punctuation except exclamation marks
    text = re.sub(r'[^a-zA-Z\s!]', '', text)

    # Lowercase all text
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back to string
    clean_text = " ".join(tokens)

    return clean_text

# Function to make prediction
def predict_mbti(text):
    # Preprocess the input text
    clean_text = preprocess_text(text)

    # Vectorize the cleaned text
    tfidf_features = tfidf_vect_ngram.transform([clean_text])

    # Make predictions
    ie_pred = xgb_ie_model.predict(tfidf_features)
    ns_pred = xgb_ns_model.predict(tfidf_features)
    tf_pred = logreg_tf_model.predict(tfidf_features)
    jp_pred = logreg_jp_model.predict(tfidf_features)

    mbti_type = f"{'I' if ie_pred[0] == 0 else 'E'}{'N' if ns_pred[0] == 0 else 'S'}{'T' if tf_pred[0] == 0 else 'F'}{'J' if jp_pred[0] == 0 else 'P'}"
    return mbti_type

# Streamlit UI
st.markdown(
    '<h1 style="color: blue;">IOD Data Science and AI:</br>Capstone Project</h1>',
    unsafe_allow_html=True
)
st.write("**Deployed By:** Gemma Cullen")

st.title('MBTI Personality Type Prediction')
input_text = st.text_area("Enter your text:")

# Button for triggering sentiment analysis
if st.button('Analyze'):
    if input_text.strip():
        # Make prediction
        prediction = predict_mbti(input_text)
        st.write(f"**MBTI Personality Type Prediction is:** {prediction}")
    else:
        st.write("Please try again.")
