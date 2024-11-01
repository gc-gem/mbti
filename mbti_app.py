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

# Dictionary mapping MBTI types to their descriptions
mbti_descriptions = {
    'ISTJ': "Quiet, serious, earn success by being thorough and dependable. Practical, matter-of-fact, realistic, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized—their work, their home, their life. Value traditions and loyalty.",
    
    'ISFJ': "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations. Thorough, painstaking, and accurate. Loyal, considerate, notice and remember specifics about people who are important to them, concerned with how others feel. Strive to create an orderly and harmonious environment at work and at home.",
    
    'INFJ': "Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to their firm values. Develop a clear vision about how best to serve the common good. Organized and decisive in implementing their vision.",
    
    'INTJ': "Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns in external events and develop long-range explanatory perspectives. When committed, organize a job and carry it through. Skeptical and independent, have high standards of competence and performance—for themselves and others.",
    
    'ISTP': "Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems. Interested in cause and effect, organize facts using logical principles, value efficiency.",
    
    'ISFP': "Quiet, friendly, sensitive, and kind. Enjoy the present moment, what's going on around them. Like to have their own space and to work within their own time frame. Loyal and committed to their values and to people who are important to them. Dislike disagreements and conflicts; don't force their opinions or values on others.",
    
    'INFP': "Idealistic, loyal to their values and to people who are important to them. Want to live a life that is congruent with their values. Curious, quick to see possibilities, can be catalysts for implementing ideas. Seek to understand people and to help them fulfill their potential. Adaptable, flexible, and accepting unless a value is threatened.",
    
    'INTP': "Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical.",
    
    'ESTP': "Flexible and tolerant, take a pragmatic approach focused on immediate results. Bored by theories and conceptual explanations; want to act energetically to solve the problem. Focus on the here and now, spontaneous, enjoy each moment they can be active with others. Enjoy material comforts and style. Learn best through doing.",
    
    'ESFP': "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and a realistic approach to their work and make work fun. Flexible and spontaneous, adapt readily to new people and environments. Learn best by trying a new skill with other people.",
    
    'ENFP': "Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see. Want a lot of affirmation from others, and readily give appreciation and support. Spontaneous and flexible, often rely on their ability to improvise and their verbal fluency.",
    
    'ENTP': "Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adept at generating conceptual possibilities and then analyzing them strategically. Good at reading other people. Bored by routine, will seldom do the same thing the same way, apt to turn to one new interest after another.",
    
    'ESTJ': "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans.",
    
    'ESFJ': "Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time. Loyal, follow through even in small matters. Notice what others need in their day-to-day lives and try to provide it. Want to be appreciated for who they are and for what they contribute.",
    
    'ENFJ': "Warm, empathetic, responsive, and responsible. Highly attuned to the emotions, needs, and motivations of others. Find potential in everyone, want to help others fulfill their potential. May act as catalysts for individual and group growth. Loyal, responsive to praise and criticism. Sociable, facilitate others in a group, and provide inspiring leadership.",
    
    'ENTJ': "Frank, decisive, assume leadership readily. Quickly see illogical and inefficient procedures and policies, develop and implement comprehensive systems to solve organizational problems. Enjoy long-term planning and goal setting. Usually well informed, well read, enjoy expanding their knowledge and passing it on to others. Forceful in presenting their ideas."
}


# Streamlit UI
st.markdown(
    """
    <style>
        .header {
            color: #4a90e2;  /* A softer blue color */
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #4a90e2; /* Underline for emphasis */
            font-family: 'Arial', sans-serif; /* Change font for better readability */
        }
        .deployed-by {
            text-align: center;
            font-size: 1.2em;
            color: #333; /* Darker color for contrast */
            margin-top: 10px;
            font-family: 'Arial', sans-serif;
        }
    </style>
    <h1 class="header">Capstone Project:<br>Typed by Text:<br>Leveraging text to predict personality type</h1>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="deployed-by">**Deployed By:** Gemma Cullen</div>', unsafe_allow_html=True)

st.title('MBTI Personality Type Prediction')
input_text = st.text_area("Enter your text:")

# Button for triggering prediction
if st.button('Analyse'):
    if input_text.strip():
        # Make prediction
        prediction = predict_mbti(input_text)
        st.write(f"**MBTI Personality Type Prediction is:** {prediction}")

        # Display the corresponding meaning
        description = mbti_descriptions.get(prediction, "Description not available.")
        st.write(f"**Meaning:** {description}")
    else:
        st.write("Please try again.")
