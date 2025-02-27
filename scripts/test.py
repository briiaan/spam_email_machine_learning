import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import messagebox

nltk.download('stopwords')

# Load the model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

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

# GUI for email classification
def classify_email():
    email_body = email_text.get("1.0", "end-1c")  # Get email body from text box
    if not email_body.strip():
        messagebox.showwarning("Input Error", "Please enter the email body.")
        return

    # Predict the label (Spam or Ham)
    prediction = predict_spam_or_ham(model, vectorizer, email_body)
    result_label.config(text=f"Prediction: {prediction}")

# Set up the GUI
root = tk.Tk()
root.title("Email Classifier")

# Load model and vectorizer
model_path = "output/model_20250219_102525.pkl"
vectorizer_path = "output/vectorizer_20250219_102525.pkl"
model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

# Set up the email input section
email_label = tk.Label(root, text="Enter the email body:")
email_label.pack(pady=10)

email_text = tk.Text(root, height=10, width=50)
email_text.pack(pady=10)

# Set up the classify button
classify_button = tk.Button(root, text="Classify Email", command=classify_email)
classify_button.pack(pady=10)

# Set up the result display
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=20)

# Start the GUI
root.mainloop()
