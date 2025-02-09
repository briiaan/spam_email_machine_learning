import os
import pandas as pd
import logging
import re

# Set up folders for cleaned datasets and logs
CLEANED_DATA_FOLDER = "sorted_datasets"
SKIPPED_DATA_FOLDER = "skipped_datasets"
LOG_FOLDER = "logs"
os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)
os.makedirs(SKIPPED_DATA_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, "data_sorting.log")),
        logging.StreamHandler()
    ]
)

# Keywords indicating spam
SPAM_KEYWORDS = ["win", "free", "prize", "cash", "offer", "congratulations", "buy now", "urgent"]

# Function to detect the most likely text column
def detect_text_column(df):
    for col in df.columns:
        # Check if column is mostly text
        if df[col].apply(lambda x: isinstance(x, str)).mean() > 0.7:  # At least 70% text
            return col
    return None

# Function to identify possible spam columns
def detect_spam_column(df):
    for col in df.columns:
        # Check if column is numeric and binary
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1}):
            return col
    return None

# Function to clean text data
def clean_text(text):
    if not isinstance(text, str):
        return "empty"  # Replace missing or invalid text with "empty"
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# Function to classify text as spam based on keywords
def classify_as_spam(text):
    if not isinstance(text, str):
        return 0  # Default to not spam for non-string text
    text = text.lower()
    if any(keyword in text for keyword in SPAM_KEYWORDS):
        return 1  # Mark as spam
    return 0  # Mark as not spam

# Function to clean and sort a dataset
def clean_and_sort_dataset(file_path):
    try:
        # Load dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            logging.warning(f"Unsupported file format: {file_path}")
            return

        initial_rows = df.shape[0]

        # Detect or validate 'text' column
        if 'text' not in df.columns:
            text_col = detect_text_column(df)
            if text_col:
                logging.info(f"Detected '{text_col}' as the text column.")
                df.rename(columns={text_col: 'text'}, inplace=True)
            else:
                logging.error(f"No suitable text column found in {file_path}. Skipping.")
                skipped_file_path = os.path.join(SKIPPED_DATA_FOLDER, os.path.basename(file_path))
                df.to_csv(skipped_file_path, index=False)
                logging.warning(f"Saved problematic dataset to {skipped_file_path}")
                return

        # Detect or validate 'spam' column
        if 'spam' not in df.columns:
            spam_col = detect_spam_column(df)
            if spam_col:
                logging.info(f"Detected '{spam_col}' as the spam column.")
                df.rename(columns={spam_col: 'spam'}, inplace=True)
            else:
                logging.warning(f"No suitable spam column found in {file_path}. Defaulting to '0' (not spam).")
                df['spam'] = df['text'].apply(classify_as_spam)  # Classify using keywords

        # Clean text column
        df['text'] = df['text'].apply(clean_text)

        # Remove rows with empty or invalid text
        valid_rows = df['text'].apply(lambda x: x != "empty").sum()
        df = df[df['text'] != "empty"]

        # Separate spam and non-spam
        spam_df = df[df['spam'] == 1]
        not_spam_df = df[df['spam'] == 0]

        # Save cleaned and sorted datasets
        cleaned_file_path = os.path.join(CLEANED_DATA_FOLDER, os.path.basename(file_path))
        spam_file_path = cleaned_file_path.replace(".csv", "_spam.csv").replace(".xlsx", "_spam.xlsx")
        not_spam_file_path = cleaned_file_path.replace(".csv", "_not_spam.csv").replace(".xlsx", "_not_spam.xlsx")

        if file_path.endswith('.csv'):
            spam_df.to_csv(spam_file_path, index=False)
            not_spam_df.to_csv(not_spam_file_path, index=False)
        elif file_path.endswith('.xlsx'):
            spam_df.to_excel(spam_file_path, index=False)
            not_spam_df.to_excel(not_spam_file_path, index=False)

        # Log usable data stats
        logging.info(f"{file_path}: Initial rows = {initial_rows}, Usable rows = {valid_rows}.")
        logging.info(f"Cleaned and sorted data saved: Spam -> {spam_file_path}, Not Spam -> {not_spam_file_path}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

# Main function to clean and sort all datasets in a folder
def clean_and_sort_datasets_in_folder(folder_path):
    supported_extensions = ['.csv', '.xlsx']
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in supported_extensions)]

    if not all_files:
        logging.error("No supported dataset files found in the folder.")
        return

    for file_path in all_files:
        clean_and_sort_dataset(file_path)

# Example usage
if __name__ == "__main__":
    folder_path = "datasets"  # Replace with your dataset folder
    clean_and_sort_datasets_in_folder(folder_path)
