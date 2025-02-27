import os
import logging
import pandas as pd
import numpy as np
import pickle
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
from tqdm import tqdm
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set up folders
FOLDERS = ["datasets", "sorted_datasets", "output", "logs", "performance_logs"]
for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "application.log")),
        logging.StreamHandler()
    ]
)

def log_to_file(message):
    with open(os.path.join("logs", "performance.log"), "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")

# Function to clean text data
def clean_text(text):
    if not isinstance(text, str):
        return "empty"
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

# Trainer: Train and evaluate the model
def train_and_evaluate_model():
    logging.info("Loading datasets...")
    log_to_file("Loading datasets...")
    files = [f for f in os.listdir("sorted_datasets") if f.endswith('.csv')]
    if not files:
        logging.error("No cleaned datasets found. Run cleaner first.")
        log_to_file("ERROR: No cleaned datasets found. Run cleaner first.")
        return
    
    dfs = [pd.read_csv(os.path.join("sorted_datasets", f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if 'text' not in df.columns or 'spam' not in df.columns:
        logging.error("Missing required columns.")
        log_to_file("ERROR: Missing required columns.")
        return
    
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]
    
    X = df['text']
    y = df['spam']
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english')
    X_transformed = vectorizer.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_transformed, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid = GridSearchCV(MultinomialNB(), param_grid, scoring='accuracy', cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    performance = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    logging.info(f"Performance: {performance}")
    log_to_file(f"Performance: {performance}")
    logging.info(f"Total Samples Used: {len(df)}")
    log_to_file(f"Total Samples Used: {len(df)}")
    logging.info(f"Spam Count: {df['spam'].sum()}, Ham Count: {len(df) - df['spam'].sum()}")
    log_to_file(f"Spam Count: {df['spam'].sum()}, Ham Count: {len(df) - df['spam'].sum()}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join("output", f"confusion_matrix_{timestamp}.png"))
    plt.close()
    log_to_file("Confusion matrix saved.")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend()
    plt.savefig(os.path.join("output", f"roc_curve_{timestamp}.png"))
    plt.close()
    log_to_file("ROC curve saved.")
    
    # Word Cloud for Spam
    spam_texts = ' '.join(df[df['spam'] == 1]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_texts)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud for Spam Emails")
    plt.savefig(os.path.join("output", f"spam_wordcloud_{timestamp}.png"))
    plt.close()
    log_to_file("Spam word cloud saved.")
    
if __name__ == "__main__":
    train_and_evaluate_model()
