from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from mlflow.tracking import MlflowClient
from tensorflow.keras.callbacks import Callback
import mlflow
from mlflow.types.schema import Schema,TensorSpec
from mlflow.models import ModelSignature
from collections import Counter



class StopOnValLossIncrease(Callback):
    def __init__(self, patience=5):
        """
        patience : nombre d'epochs sans amélioration avant d'arrêter
        """
        super(StopOnValLossIncrease, self).__init__()
        self.patience = patience
        self.best_val_loss = float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")

        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss  # Sauvegarde de la meilleure valeur atteinte
            self.wait = 0
        else:
            self.wait += 1
            print(f"⚠️ Augmentation de val_loss détectée ({val_loss}). Patience: {self.wait}/{self.patience}")

            if self.wait >= self.patience:
                print("⛔ Arrêt de l'entraînement pour éviter l'overfitting.")
                self.model.stop_training = True

stop_on_val_loss = StopOnValLossIncrease(patience=5)

# Étape 1 : Récupérer les données depuis Elasticsearch
print("Récupération des données depuis Elasticsearch...")
es = Elasticsearch("http://elasticsearch:9200")
index_name = "github_repos"
query = {"query": {"match_all": {}}}
response = es.search(index=index_name, body=query, size=1000)
data = [hit["_source"] for hit in response["hits"]["hits"]]

print(f"Nombre de documents récupérés: {len(data)}")

# Étape 2 : Préparer les données
print("Préparation des données...")
df = pd.DataFrame(data)

# Filtrer les lignes où la colonne "topics" est vide
df = df[df["topics"].apply(lambda x: len(x) > 0)]

print(f"Nombre de lignes après filtrage: {len(df)}")

# Générer la liste des labels uniques à partir des topics
all_labels = [label for topics in df["topics"] for label in topics]
label_counts = Counter(all_labels)

# Afficher les 50 labels les plus fréquents
most_common_labels = label_counts.most_common(50)
print(f"50 labels les plus fréquents: {most_common_labels}")

# Sélectionner les 50 labels les plus fréquents
labels_to_keep = [label for label, count in most_common_labels]

print(f"Nombre de labels conservés: {len(labels_to_keep)}")

# Créer une colonne binaire pour chaque label
for label in labels_to_keep:
    df[label] = df["topics"].apply(lambda topics: 1 if label in topics else 0)

# Fonction pour vérifier et aplatir les embeddings
def flatten_embedding(embedding):
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        # Aplatir une liste de listes
        return [item for sublist in embedding for item in sublist]
    elif isinstance(embedding, list):
        return embedding
    else:
        return []

# Appliquer le flattening aux colonnes d'embeddings
print("Aplatissement des embeddings...")
df["description_embedding"] = df["description_embedding"].apply(flatten_embedding)
df["languages_embedding"] = df["languages_embedding"].apply(flatten_embedding)
df["repo_name_embedding"] = df["repo_name_embedding"].apply(flatten_embedding)

# Déterminer la longueur maximale des embeddings
embedding_length = max(
    df["description_embedding"].apply(len).max(),
    df["languages_embedding"].apply(len).max(),
    df["repo_name_embedding"].apply(len).max()
)

print(f"Longueur maximale des embeddings: {embedding_length}")

# Fonction de padding pour uniformiser la taille des embeddings
def pad_embedding(embedding, length=embedding_length):
    return embedding + [0] * (length - len(embedding))

df["description_embedding"] = df["description_embedding"].apply(lambda x: pad_embedding(x, embedding_length))
df["languages_embedding"] = df["languages_embedding"].apply(lambda x: pad_embedding(x, embedding_length))
df["repo_name_embedding"] = df["repo_name_embedding"].apply(lambda x: pad_embedding(x, embedding_length))

# Pour construire X, on va concaténer les trois embeddings pour chaque exemple
X_desc = np.array(df["description_embedding"].tolist(), dtype=np.float32)
X_lang = np.array(df["languages_embedding"].tolist(), dtype=np.float32)
X_repo = np.array(df["repo_name_embedding"].tolist(), dtype=np.float32)

# Concatenation horizontale : chaque exemple aura une taille = embedding_length * 3
X = np.concatenate([X_desc, X_lang, X_repo], axis=1)

# y : matrice binaire des labels
y = df[labels_to_keep].values.astype(np.float32)

# Étape 3 : Diviser les données
print("Division des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Démarrer une nouvelle expérience avec MLflow
mlflow.set_tracking_uri("http://mlflow:5000")  # Pointage vers l'instance MLflow
mlflow.keras.autolog()  # Active l'autologging pour Keras
with mlflow.start_run():  # Démarrage d'une nouvelle run

    print("Création du modèle...")
    # Étape 4 : Créer le modèle
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(labels_to_keep), activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Enregistrement des paramètres dans MLflow...")
    # Enregistrement des paramètres du modèle dans MLflow
    mlflow.log_param("input_shape", (5,))
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 32)
    mlflow.autolog()

    artifact_file = "labels.json"
    labels_dict = {"labels": labels_to_keep}
    mlflow.log_dict(labels_dict, artifact_file)

    print("Entraînement du modèle...")
    # Étape 5 : Entraîner le modèle
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    print("Évaluation du modèle...")
    # Étape 6 : Évaluer le modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    print("Sauvegarde du modèle dans MLflow...")
    # Étape 7 : Sauvegarder le modèle dans MLflow avec signature et exemple d'entrée

    input_example = X_train[0:1].astype(np.float32)  # Exemple d'entrée

    # Définir la signature manuellement en utilisant TensorSpec
    input_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, X.shape[1]))])
    output_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, len(labels_to_keep)))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.keras.log_model(model, "model", signature=signature)

    print("Modèle enregistré.")

client = MlflowClient()

# Rechercher les runs de l'expérience, triées par "accuracy" décroissante
runs = client.search_runs(experiment_ids=["0"], order_by=["metrics.accuracy DESC"], max_results=1)

import mlflow
import json

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
        exit()

    # Faire des prédictions avec le modèle chargé
    predictions = loaded_model.predict(X_test)
    print(f"Forme des prédictions: {predictions.shape}")  # Vérifier la forme des prédictions

    try:
        # Créer l'URI de l'artefact avec le run_id correct
        artifact_uri = f"mlflow-artifacts:/0/{best_run_id}/artifacts/labels.json"

        # Télécharger l'artefact
        download_path = mlflow.artifacts.download_artifacts(artifact_uri)
        print(f"Artefact téléchargé depuis {download_path}.")

    except Exception as e:
        print(f"Erreur lors du téléchargement de l'artefact: {e}")
        exit()

    # Charger les labels depuis le fichier JSON téléchargé
    try:
        with open(download_path, "r") as f:
            labels_data = json.load(f)
        labels = labels_data["labels"]
        print(f"Labels récupérés: {labels}")
    except Exception as e:
        print(f"Erreur lors du chargement des labels: {e}")
        exit()

    # Conversion des prédictions en labels
    predicted_labels = []
    for i in range(len(predictions)):
        predicted = []
        for j in range(len(predictions[i])):
            if predictions[i][j] > 0.5:  # Seuil pour attribuer le label
                predicted.append(labels[j])
        predicted_labels.append(predicted)

    # Affichage des résultats
    for i, label_set in enumerate(predicted_labels):
        print(f"Prédictions pour l'exemple {i + 1}: {label_set}")
else:
    print("Aucun run trouvé dans l'expérience.")

