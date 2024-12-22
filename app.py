from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import random
nltk.download('punkt_tab')

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS for all routes, but restrict it to your frontend (localhost:3000)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Load pre-trained model and necessary files
model = joblib.load('public/bestmodel.joblib')
tfidf_vectorizer = joblib.load('public/tfidf_vectorizer.joblib')
stopwords_list = joblib.load('public/stopwords.pkl')
stemmer = joblib.load('public/porter_stemmer.pkl')

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing functions
def convert_lowercase(text):
    return text.lower()

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    return ' '.join([word for word in words if word not in stopwords_list])

def perform_stemming(text):
    words = nltk.word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words])

def preprocess_text(text):
    text = convert_lowercase(text)
    text = remove_url(text)
    text = remove_punc(text)
    text = remove_stopwords(text)
    text = perform_stemming(text)
    return text

# Define an array of random labels for both "Anxiety/Depression" and "No Anxiety/Depression"
anxiety_labels = [
    "It looks like you're experiencing anxiety or depression.",
    "Your response indicates possible anxiety or depression.",
    "Based on your input, you might be dealing with anxiety or depression.",
    "The analysis suggests that you may have anxiety or depression."
]

no_anxiety_labels = [
    "Your response indicates that you're not experiencing anxiety or depression.",
    "Based on your input, you don't seem to have anxiety or depression.",
    "It looks like you're not dealing with anxiety or depression at the moment.",
    "The analysis suggests that you're free from anxiety or depression."
]

# Define the API route for prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    # Preprocess the input text
    clean_text = preprocess_text(text)
    
    # Vectorize the text and make prediction
    text_vectorized = tfidf_vectorizer.transform([clean_text]).toarray()
    prediction = model.predict(text_vectorized)[0]
    
    # Map prediction to label
    if prediction == 1:
        label = random.choice(anxiety_labels)  # Select a random anxiety/depression message
    else:
        label = random.choice(no_anxiety_labels)  # Select a random no anxiety/depression message
    
    return jsonify({"result": label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
