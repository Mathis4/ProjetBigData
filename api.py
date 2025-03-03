from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import requests

app = FastAPI()

# === Partie optionnelle MLflow (désactivée) ===
# import mlflow
# from mlflow.tracking import MlflowClient
# import mlflow.keras
# import pandas as pd
#
# # Configurer l'URI de tracking pour pointer vers le serveur MLflow accessible depuis ce conteneur
# # mlflow.set_tracking_uri("http://adresse_du_serveur_mlflow:5000")
#
# # Initialiser le client MLflow
# # client = MlflowClient()
#
# # Définir l'expérience à utiliser (par exemple "Default")
# # experiment_name = "Default"
# # experiment = client.get_experiment_by_name(experiment_name)
# # experiment_id = experiment.experiment_id if experiment is not None else "0"
#
# # Rechercher le run avec la meilleure accuracy
# # runs = client.search_runs(
# #     experiment_ids=[experiment_id],
# #     order_by=["metrics.accuracy DESC"],
# #     max_results=1
# # )
#
# # if runs:
# #     best_run = runs[0]
# #     best_run_id = best_run.info.run_id
# #     best_accuracy = best_run.data.metrics.get("accuracy")
# #     print(f"Chargement du modèle du run '{best_run_id}' avec accuracy: {best_accuracy}")
# #
# #     # Charger le modèle depuis MLflow
# #     model_uri = f"runs:/{best_run_id}/model"
# #     loaded_model = mlflow.keras.load_model(model_uri)

# === Chargement du modèle Keras sauvegardé ===
loaded_model = load_model('my_model.keras')
labels = ["Web Development", "Data Analysis", "Blockchain"]  # Exemple de labels possibles

# === Définition des modèles de données ===
class RepoInput(BaseModel):
    repo_url: str

class TopicSuggestionInput(BaseModel):
    title: str
    technologies: str
    readme_content: str

# === Endpoint pour extraire les topics d'un dépôt GitHub ===
@app.get("/github-topics/")
def get_github_topics(repo_url: str):
    api_url = f"https://api.github.com/repos/{repo_url}/topics"
    headers = {"Accept": "application/vnd.github.mercy-preview+json"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        topics = response.json().get("names", [])
        return {"topics": topics}
    else:
        raise HTTPException(status_code=response.status_code, detail="Impossible de récupérer les topics depuis GitHub")

# === Endpoint pour suggérer des topics à partir des caractéristiques du projet ===
@app.post("/suggest-topics-details/")
def suggest_topics(input_data: TopicSuggestionInput):
    # Préparer les features à partir de trois champs
    input_features = [
        len(input_data.title.split()),
        len(input_data.technologies.split()),
        len(input_data.readme_content.split())
    ]
    input_features = np.array([input_features])
    
    # Faire la prédiction avec le modèle Keras
    predictions = loaded_model.predict(input_features)
    
    # Convertir les prédictions en labels en appliquant un seuil de classification (0.3)
    predicted_labels = [labels[i] for i in range(len(predictions[0])) if predictions[0][i] > 0.3]
    return {"suggested_topics": predicted_labels}

# === Endpoint de test ===
@app.get("/test/")
def test_service():
    return {"status": "FastAPI fonctionne parfaitement !"}
