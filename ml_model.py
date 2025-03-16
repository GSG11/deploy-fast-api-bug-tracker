import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure necessary NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

REQUIRED_COLUMNS = {"Summary", "Description"}

# Load dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        print("Error: CSV file not found!")
        return None
    
    df = pd.read_csv(file_path)
    
    # Ensure at least one required column exists
    available_columns = REQUIRED_COLUMNS.intersection(df.columns)
    if not available_columns:
        print("Error: No relevant text columns found!")
        print("Expected at least one of:", REQUIRED_COLUMNS)
        print("Found columns:", df.columns.tolist())
        return None
    
    # Handle missing columns
    df["Summary"] = df["Summary"].fillna("") if "Summary" in df.columns else ""
    df["Description"] = df["Description"].fillna("") if "Description" in df.columns else ""
    
    # Combine only available columns
    if "Summary" in df.columns and "Description" in df.columns:
        df["text"] = df["Summary"].astype(str) + " " + df["Description"].astype(str)
    else:
        df["text"] = df["Description"].astype(str) if "Description" in df.columns else df["Summary"].astype(str)
    
    # Drop rows with empty text
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    
    if df.empty:
        print("Error: CSV file is empty or contains only NaN values.")
        return None
    
    return df

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Compute similarity
def compute_similarity(df):
    df["processed_text"] = df["text"].apply(preprocess_text)
    
    # Select subset for faster computation
    df_sample = df.head(1000)  # Use first 1000 rows
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_sample["processed_text"])
    
    if tfidf_matrix.shape[0] == 0:
        print("Warning: No data available for vectorization.")
        return None, None
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix, df_sample

# Identify duplicate bug reports
def find_duplicates(similarity_matrix, df_sample, threshold=0.8):
    duplicate_pairs = [(i, j, similarity_matrix[i, j])
                       for i in range(len(df_sample))
                       for j in range(i + 1, len(df_sample))
                       if similarity_matrix[i, j] > threshold]
    
    if duplicate_pairs:
        print("\nPotential Duplicate Bug Reports:")
        for i, j, sim in duplicate_pairs:
            print(f"Bug {i} and Bug {j} are {sim:.2f}% similar")
    else:
        print("\nNo duplicate reports detected above threshold.")

# Main function
def main():
    file_path = "eclipse_platform.csv"  # Update with actual file path
    df = load_data(file_path)
    
    if df is not None:
        similarity_matrix, df_sample = compute_similarity(df)
        if similarity_matrix is not None:
            find_duplicates(similarity_matrix, df_sample)

if __name__ == "__main__":
    main()