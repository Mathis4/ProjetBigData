from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import tensorflow as tf
import numpy as np

# Charger le modèle de machine learning
model = tf.keras.models.load_model("my_model.keras")

app = FastAPI()

class GitHubProject(BaseModel):
    repo_url: str

def fetch_project_info(repo_url: str):
    headers = {"Accept": "application/vnd.github.v3+json"}
    owner, repo = repo_url.strip("/").split("/")[-2:]
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Impossible de récupérer les infos du projet.")
    
    repo_data = response.json()
    title = repo_data.get("name", "")
    description = repo_data.get("description", "")
    technologies = repo_data.get("language", "")
    
    return title, description, technologies

def predict_topics(title, description, technologies):
    # Exemple d'entrée pour le modèle (convertir les inputs en features numériques)
    input_data = np.array([len(title), len(description), len(technologies)]).reshape(1, -1)
    prediction = model.predict(input_data)
    suggested_topics = ["Topic1", "Topic2", "Topic3"]  # Exemple de topics à retourner
    return suggested_topics

@app.post("/suggest-topics/")
def suggest_topics(project: GitHubProject):
    title, description, technologies = fetch_project_info(project.repo_url)
    topics = predict_topics(title, description, technologies)
    return {"suggested_topics": topics}

