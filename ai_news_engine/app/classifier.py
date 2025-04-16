import joblib
import re
import os

# Get path to current file (classifier.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for the model and vectorizer
model_path = os.path.join(base_dir, "model", "news_classifier_model.pkl")
vectorizer_path = os.path.join(base_dir, "model", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction function
def predict_category(text):
    try:
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        return prediction
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Uncategorized"  # Fallback in case of errors
