from fastapi import FastAPI, File, UploadFile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

app = FastAPI(title="AI Bug Tracker", description="Detects duplicate bug reports using NLP.", version="1.0")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

REQUIRED_COLUMNS = {"Summary", "Description"}

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Compute similarity
def compute_similarity(df):
    df["processed_text"] = df["text"].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
    
    if tfidf_matrix.shape[0] == 0:
        return None
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

# Identify duplicate bug reports
def find_duplicates(similarity_matrix, df, threshold=0.8):
    duplicate_pairs = [
        {"Bug1": i, "Bug2": j, "Similarity": round(similarity_matrix[i, j], 2)}
        for i in range(len(df))
        for j in range(i + 1, len(df))
        if similarity_matrix[i, j] > threshold
    ]
    
    return {"duplicates": duplicate_pairs}

# FastAPI route to upload and process CSV
@app.post("/detect_duplicates")
async def detect_duplicates(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))

        # Ensure relevant columns exist
        available_columns = REQUIRED_COLUMNS.intersection(df.columns)
        if not available_columns:
            return {"error": "CSV must contain at least one of these columns: Summary, Description"}
        
        df["Summary"] = df["Summary"].fillna("") if "Summary" in df.columns else ""
        df["Description"] = df["Description"].fillna("") if "Description" in df.columns else ""
        df["text"] = df["Summary"] + " " + df["Description"]
        df = df[df["text"].str.strip() != ""].reset_index(drop=True)

        if df.empty:
            return {"error": "CSV is empty or contains only NaN values."}

        similarity_matrix = compute_similarity(df)
        if similarity_matrix is None:
            return {"error": "No valid data for comparison."}
        
        return find_duplicates(similarity_matrix, df)

    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Bug Tracker API"}

