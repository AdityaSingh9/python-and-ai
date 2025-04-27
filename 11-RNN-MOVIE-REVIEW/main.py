# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('/Users/adityakumarsingh/python/11-RNN-MOVIE-REVIEW/simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit App
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", page_icon="🎥", layout="centered")

# Apply a custom style
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6c6c6c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 10em;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.markdown('<div class="title">🎬 IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter a movie review and find out if it\'s Positive or Negative!</div>', unsafe_allow_html=True)

    # Input box
    user_input = st.text_area('✍️ Your Movie Review', placeholder="Type your review here...", height=200)

    # Predict button
    if st.button('Analyze Sentiment'):
        if user_input.strip() == "":
            st.warning('⚠️ Please enter a movie review first.')
        else:
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            sentiment = '👍 Positive' if prediction[0][0] > 0.5 else '👎 Negative'
            score = float(prediction[0][0])

            # Display the results nicely
            st.subheader('Result:')
            st.success(f"**Sentiment:** {sentiment}")
            # st.info(f"**Prediction Confidence:** {score:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)
