from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import requests

app = FastAPI()

# Charger le modèle Keras sauvegardé
loaded_model = load_model('my_model.keras')
labels = ["Web Development", "Data Analysis", "Blockchain"]  # Exemple de labels possibles


class RepoInput(BaseModel):
    repo_url: str

class TopicSuggestionInput(BaseModel):
    title: str
    technologies: str
    readme_content: str
    feature4: str
    feature5: str


@app.get("/github-topics/")
def get_github_topics(repo_url: str):
    # Extraction simulée des topics d'un projet GitHub via l'API GitHub
    api_url = f"https://api.github.com/repos/{repo_url}/topics"
    headers = {"Accept": "application/vnd.github.mercy-preview+json"}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        topics = response.json().get("names", [])
        return {"topics": topics}
    else:
        raise HTTPException(status_code=response.status_code, detail="Impossible de récupérer les topics depuis GitHub")


@app.post("/suggest-topics-details/")
def suggest_topics(input_data: TopicSuggestionInput):
    # Préparer les données d'entrée pour le modèle
    input_features = [
        len(input_data.title.split()),
        len(input_data.technologies.split()),
        len(input_data.readme_content.split()),
        len(input_data.feature4.split()),
        len(input_data.feature5.split())
    ]
    input_features = np.array([input_features])

    # Faire une prédiction avec le modèle Keras
    predictions = loaded_model.predict(input_features)

    # Conversion des prédictions en labels
    predicted_labels = []
    for i in range(len(predictions[0])):
        if predictions[0][i] > 0.3:  # Seuil de classification
            predicted_labels.append(labels[i])

    # Retourner les labels prédits
    return {"suggested_topics": predicted_labels}


@app.get("/test/")
def test_service():
    return {"status": "FastAPI fonctionne parfaitement !"}
