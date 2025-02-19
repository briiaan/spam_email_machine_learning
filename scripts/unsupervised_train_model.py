import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk

# 🔹 Ensure NLTK stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

print("✅ Stopwords loaded successfully.")

# 📂 **Folder containing batch DOCX files**
batch_folder = r"C:\Users\eric_\Desktop\Senior Project\cluster folder\datasets\docx_output"

# 📂 **Folder where models & vectorizer will be saved**
output_folder = os.path.join(batch_folder, "saved_models")
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# 📌 **Extract text from DOCX files**
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error extracting text from {file_path}: {e}")
        return ""

# 📌 **Find all batch DOCX files**
batch_files = [f for f in os.listdir(batch_folder) if f.endswith('.docx')]
if not batch_files:
    print("❌ No batch DOCX files found. Exiting...")
    exit()

print(f"✅ Found {len(batch_files)} batch files. Processing...")

# 📌 **Load all batch files into a DataFrame**
df_list = []
for file in batch_files:
    file_path = os.path.join(batch_folder, file)
    text = extract_text_from_docx(file_path)
    df_list.append({"filename": file, "text": text})

df = pd.DataFrame(df_list)

# 📌 **Clean the extracted text**
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters, numbers
    text = text.lower().strip()  # Convert to lowercase
    words = re.split(r'\s+', text)  # Tokenize text
    words = [word for word in words if word and word not in stop_words]  # Remove stopwords
    return " ".join(words)  # Reconstruct text

df['cleaned_text'] = df['text'].apply(clean_text)

print(f"✅ Text cleaning complete. Total processed records: {df.shape[0]}")

# ✅ Ensure there's data before clustering
if df.empty:
    print("❌ No valid data to process. Exiting...")
    exit()

# 🔹 Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_transformed = vectorizer.fit_transform(df['cleaned_text'])

print("✅ TF-IDF transformation complete.")

# 🔹 Apply K-Means Clustering
num_clusters = 3  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
df['kmeans_cluster'] = kmeans.fit_predict(X_transformed)

print("✅ K-Means clustering complete.")

# 🔹 Apply DBSCAN for alternative clustering
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
df['dbscan_cluster'] = dbscan.fit_predict(X_transformed.toarray())

print("✅ DBSCAN clustering complete.")

# 🔹 Save the trained models and vectorizer
kmeans_model_path = os.path.join(output_folder, "unsupervised_kmeans_model.pkl")
dbscan_model_path = os.path.join(output_folder, "unsupervised_dbscan_model.pkl")
vectorizer_path = os.path.join(output_folder, "vectorizer.pkl")

with open(kmeans_model_path, "wb") as model_file:
    pickle.dump(kmeans, model_file)

with open(dbscan_model_path, "wb") as model_file:
    pickle.dump(dbscan, model_file)

with open(vectorizer_path, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"\n✅ Models and vectorizer saved successfully in: {output_folder}")
print(f"📁 K-Means Model: {kmeans_model_path}")
print(f"📁 DBSCAN Model: {dbscan_model_path}")
print(f"📁 TF-IDF Vectorizer: {vectorizer_path}")

# 📊 **Visualize K-Means Clustering**
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed.toarray())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['kmeans_cluster'], palette="coolwarm", s=100, edgecolor="black")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering Visualization")
plt.savefig(os.path.join(output_folder, "kmeans_clustering.png"))
plt.close()

# 📊 **Visualize DBSCAN Clustering**
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['dbscan_cluster'], palette="viridis", s=100, edgecolor="black")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("DBSCAN Clustering Visualization")
plt.savefig(os.path.join(output_folder, "dbscan_clustering.png"))
plt.close()

print(f"\n✅ Clustering visualizations saved in: {output_folder}")
