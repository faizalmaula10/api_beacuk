from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

nltk.download('stopwords')

app = FastAPI()

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing tools
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Input schema
class CommentRequest(BaseModel):
    comment: str

# Clean text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

@app.get("/")
def home():
    return {"message": "Sentiment API Running"}

@app.post("/predict")
def predict(data: CommentRequest):
    clean = preprocess(data.comment)

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    classes = model.classes_

    prob_result = {
        classes[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(classes))
    }

    confidence = round(max(probabilities) * 100, 2)

    return {
        "comment": data.comment,
        "clean_text": clean,
        "prediction": prediction,
        "confidence_percent": confidence,
        "probabilities": prob_result
    }