# app/classifier.py
import joblib
import re
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to current file
model_path = os.path.join(base_dir, "model", "svm_model.pkl")
model = joblib.load(model_path)

vectorizer = joblib.load("app/model/vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_category(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return prediction
