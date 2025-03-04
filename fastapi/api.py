import mlflow
import json
import numpy as np
import requests
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
import mlflow.keras
from fastapi import FastAPI, HTTPException
from generate_embeddings import generate_embedding, flatten_embedding, pad_embedding

app = FastAPI()

def load_best_model():
    # Configurer l'URI de tracking pour pointer vers le serveur MLflow accessible depuis ce conteneur
    mlflow.set_tracking_uri("http://mlflow:5000")

    client = MlflowClient()
    # Rechercher les runs de l'expérience, triées par "accuracy" décroissante
    runs = client.search_runs(experiment_ids=["0"], order_by=["metrics.accuracy DESC"], max_results=1)

    if runs:

        best_run = runs[0]
        best_run_id = best_run.info.run_id
        best_accuracy = best_run.data.metrics.get("accuracy")
        print(f"Chargement du modèle avec run_id: {best_run_id} et accuracy: {best_accuracy}")

        # Vérification de l'URI du modèle
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Modèle URI: {model_uri}")

        # Charger le modèle depuis MLflow
        try:
            loaded_model = mlflow.keras.load_model(model_uri)
            print("Modèle chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return None, None

        try:
            # Créer l'URI de l'artefact avec le run_id correct
            artifact_uri = f"mlflow-artifacts:/0/{best_run_id}/artifacts/labels.json"

            # Télécharger l'artefact
            download_path = mlflow.artifacts.download_artifacts(artifact_uri)
            print(f"Artefact téléchargé depuis {download_path}.")
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'artefact: {e}")
            return loaded_model, None  # Retourne le modèle même si les labels ne sont pas chargés

        # Charger les labels depuis le fichier JSON téléchargé
        try:
            with open(download_path, "r") as f:
                labels_data = json.load(f)
            labels = labels_data["labels"]
            print(f"Labels récupérés: {labels}")
        except Exception as e:
            print(f"Erreur lors du chargement des labels: {e}")
            return loaded_model, None

        return loaded_model, labels
    else:
        print("Aucun run trouvé.")
        return None, None




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
    # Charger le meilleur modèle et les labels
    loaded_model, labels = load_best_model()

    if loaded_model is None or labels is None:
        return {"error": "Impossible de charger le modèle ou les labels."}

    # Générer les embeddings à partir des textes
    title_emb = generate_embedding(input_data.title)
    tech_emb = generate_embedding(input_data.technologies)
    readme_emb = generate_embedding(input_data.readme_content)

    # Aplatir les embeddings
    title_emb = flatten_embedding(title_emb)
    tech_emb = flatten_embedding(tech_emb)
    readme_emb = flatten_embedding(readme_emb)

    # Déterminer la longueur maximale des embeddings (peut être déterminée dynamiquement ou fixée)
    embedding_length = max(len(title_emb), len(tech_emb), len(readme_emb))

    # Appliquer le padding pour uniformiser les tailles
    title_emb = pad_embedding(title_emb, embedding_length)
    tech_emb = pad_embedding(tech_emb, embedding_length)
    readme_emb = pad_embedding(readme_emb, embedding_length)

    # Concaténation des embeddings
    input_features = np.concatenate([title_emb, tech_emb, readme_emb]).reshape(1, -1)

    # Prédiction avec le modèle Keras
    predictions = loaded_model.predict(input_features)

    # Convertir les scores en labels avec un seuil de 0.3
    predicted_labels = [labels[i] for i in range(len(predictions[0])) if predictions[0][i] > 0.3]

    return {"suggested_topics": predicted_labels}


# === Endpoint de test ===
@app.get("/test/")
def test_service():
    return {"status": "FastAPI fonctionne parfaitement !"}