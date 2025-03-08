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
MODEL_PATH = "./model.pkl"
VECTORIZER_PATH = "./vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Trained model or vectorizer not found. Please train the model first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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

def clean_text(text):
    """Preprocess text: lowercase, remove special characters, stopwords, and lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def classify_text(text):
    """Classify text as spam (1) or not spam (0)."""
    text_cleaned = clean_text(text)
    if not text_cleaned:
        return 0  # Default to not spam for empty or invalid text
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return int(prediction[0])

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
