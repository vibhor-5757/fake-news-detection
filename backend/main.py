from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import spacy

# Load spaCy
nlp = spacy.load("en_core_web_md")

models = {
    "countvec": {
        "model": pickle.load(open("./models/model_countvec.pkl", "rb")),
        "vectorizer": pickle.load(open("./models/vectorizer_countvec.pkl", "rb")),
        "scaler": pickle.load(open("./models/scaler_countvec.pkl", "rb"))
    }
    # ,
    # "tfidf": {
    #     "model": pickle.load(open("./models/model_tfidf.pkl", "rb")),
    #     "vectorizer": pickle.load(open("./models/vectorizer_tfidf.pkl", "rb")),
    #     "scaler": pickle.load(open("./models/scaler_tfidf.pkl", "rb"))
    # },
    # "spacy": {
    #     "model": pickle.load(open("./models/model_spacy.pkl", "rb")),
    #     "scaler": pickle.load(open("./models/scaler_spacy.pkl", "rb"))
    # }
}

# Request format
class Article(BaseModel):
    text: str

app = FastAPI()

@app.post("/predict/{model_name}")
def predict_news(model_name: str, article: Article):
    text = article.text

    if model_name not in models:
        return {"error": "Model not found"}

    model_data = models[model_name]
    model = model_data["model"]
    scaler = model_data["scaler"]
    vectorizer = model_data["vectorizer"]

    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    text = ' '.join(tokens)

    vector = vectorizer.transform([text]).toarray()

    vector = scaler.transform(vector)
    prob = model.predict_proba(vector)[0][1]  # Assuming '1' is for 'fake'
    return {
        "label": "Fake" if prob > 0.5 else "Real",
        "confidence_score": round(prob * 100, 2)
    }
