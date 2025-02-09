import os
import logging
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import re
import datetime
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Set up folders
DATASETS_FOLDER = "datasets"
CLEANED_DATA_FOLDER = "sorted_datasets"
OUTPUT_FOLDER = "output"
LOG_FOLDER = "logs"
os.makedirs(DATASETS_FOLDER, exist_ok=True)
os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, "application.log")),
        logging.StreamHandler()
    ]
)

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
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Cleaner: Process raw datasets
def clean_and_sort_datasets():
    supported_extensions = ['.csv', '.xlsx']
    files = [f for f in os.listdir(DATASETS_FOLDER) if any(f.endswith(ext) for ext in supported_extensions)]

    for file in tqdm(files, desc="Cleaning datasets"):
        file_path = os.path.join(DATASETS_FOLDER, file)
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                logging.warning(f"Unsupported file format: {file}")
                continue

            if 'text' not in df.columns or 'spam' not in df.columns:
                logging.warning(f"Skipping {file}: Missing 'text' or 'spam' column.")
                continue

            df['text'] = df['text'].apply(clean_text)
            df['spam'] = pd.to_numeric(df['spam'], errors='coerce').fillna(0).astype(int)
            valid_rows = df[df['text'] != "empty"]

            cleaned_file_path = os.path.join(CLEANED_DATA_FOLDER, file)
            valid_rows.to_csv(cleaned_file_path, index=False)
            logging.info(f"Cleaned data saved to {cleaned_file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

# Trainer: Train and evaluate the model
def train_and_evaluate_model():
    logging.info(f"Loading datasets from {CLEANED_DATA_FOLDER}")
    files = [f for f in os.listdir(CLEANED_DATA_FOLDER) if f.endswith('.csv')]
    if not files:
        logging.error("No cleaned datasets found. Please run the cleaner first.")
        return

    dfs = []
    for file in files:
        file_path = os.path.join(CLEANED_DATA_FOLDER, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
    
    if not dfs:
        logging.error("No valid datasets for training.")
        return

    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded combined dataset of size: {df.shape}")

    if 'text' not in df.columns or 'spam' not in df.columns:
        logging.error("Required columns 'text' and 'spam' missing.")
        return

    # Preprocess and vectorize
    X = df['text']
    y = df['spam']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_transformed = vectorizer.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_transformed, y)

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid = GridSearchCV(MultinomialNB(), param_grid, scoring='accuracy', cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    logging.info(f"Best parameters: {grid.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Save model and vectorizer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(OUTPUT_FOLDER, f"model_{timestamp}.pkl")
    vectorizer_path = os.path.join(OUTPUT_FOLDER, f"vectorizer_{timestamp}.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(best_model, model_file)
    with open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    logging.info(f"Model and vectorizer saved: {model_path}, {vectorizer_path}")

    # Generate plots
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"confusion_matrix_{timestamp}.png"))
    plt.close()

    # ROC Curve
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"roc_curve_{timestamp}.png"))
    plt.close()
    logging.info(f"ROC curve saved.")

    # Word Cloud
    spam_texts = ' '.join(df[df['spam'] == 1]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_texts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud for Spam Emails")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"spam_wordcloud_{timestamp}.png"))
    plt.close()
    logging.info("Word cloud saved.")

# Main workflow
if __name__ == "__main__":
    clean_and_sort_datasets()  # Step 1: Clean datasets
    train_and_evaluate_model()  # Step 2: Train and evaluate
