from flask import Flask, request, jsonify
import os
import logging
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load pre-trained model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Set up logging
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, "spam_detection.log")),
        logging.StreamHandler()
    ]
)

# Clean the input text
def clean_text(text):
    if not isinstance(text, str):
        return "empty"
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Predict using the trained model
def predict_spam_or_ham(model, vectorizer, email_body):
    cleaned_email = clean_text(email_body)
    email_vector = vectorizer.transform([cleaned_email])
    prediction = model.predict(email_vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """Receive text and return spam classification."""
    try:
        if request.method == 'POST':
            data = request.get_json()
            text = data.get("text", "")
        else:  # GET request
            text = request.args.get("text", "")

        result = classify_text(text)
        return jsonify({"spam": result})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)