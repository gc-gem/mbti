import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
tfidf_vect_ngram = joblib.load('tfidf_vect_ngram.pkl')
xgb_ie_model = joblib.load('xgb_ie_model.pkl')
xgb_ns_model = joblib.load('xgb_ns_model.pkl')
logreg_tf_model = joblib.load('logreg_tf_model.pkl')
logreg_jp_model = joblib.load('logreg_jp_model.pkl')

# Function to preprocess text input
def preprocess_text(text):
    tfidf_vect = tfidf_vect_ngram.transform([text])
    return tfidf_vect

# Function to make prediction
def predict_mbti(text):
    ie_pred = xgb_ie_model.predict([text])
    ns_pred = xgb_ns_model.predict([text])
    tf_pred = logreg_tf_model.predict([text])
    jp_pred = logreg_jp_model.predict([text])

    mbti_type = f"{'I' if ie_pred == 0 else 'E'}{'N' if ns_pred == 0 else 'S'}{'T' if tf_pred == 0 else 'F'}{'J' if jp_pred == 0 else 'P'}"
    return mbti_type

# Streamlit UI
st.markdown(
    '<h1 style="color: orange;">IOD Data Science and AI: Capstone Project</h1>',
    unsafe_allow_html=True
)
st.write("**Deployed By:** Gemma Cullen")

st.title('MBTI Personality Type Prediction')
input_text = st.text_area("Enter your text:")

# Button for triggering sentiment analysis
if st.button('Analyze'):
    if input_text.strip():
        # Preprocess user input
        features = preprocess_text(input_text)
        # Make prediction
        prediction = predict_mbti.predict(features)
        st.write(f"**MBTI Personality Type Prediction is:** {prediction}")
    else:
        st.write("Please try again.")