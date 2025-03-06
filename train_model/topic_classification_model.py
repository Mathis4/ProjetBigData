import os
import threading
import time
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from collections import Counter
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


def train_model():
    """Fonction d'entraînement du modèle."""

    # Étape 1 : Récupérer les données depuis Elasticsearch
    print("Récupération des données depuis Elasticsearch...")
    es = Elasticsearch("http://elasticsearch:9200")
    index_name = "github_repos"
    query = {"query": {"match_all": {}}}
    response = es.search(index=index_name, body=query, size=9999)
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

    # Étape 3 : Diviser les données en ensembles d'entraînement et de test
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


def get_dataset_from_elasticsearch():
    """Récupère les données depuis Elasticsearch."""
    print("Récupération des données depuis Elasticsearch...")
    es = Elasticsearch("http://elasticsearch:9200")

    index_name = "github_repos"
    query = {"query": {"match_all": {}}}
    response = es.search(index=index_name, body=query, size=1000)

    data = [hit["_source"] for hit in response["hits"]["hits"]]
    print(f"Nombre de documents récupérés: {len(data)}")

    return data


class DatasetScheduler:
    def __init__(self, interval=600):
        """
        interval : temps en secondes entre chaque vérification (600s = 10 minutes)
        """
        self.interval = interval
        self.previous_data_count = None
        self.running = True

    def get_dataset_from_elasticsearch(self):
        """Récupérer le dataset depuis Elasticsearch."""
        try:
            es = Elasticsearch("http://elasticsearch:9200")
            if not es.ping():
                print("⚠️ Impossible de se connecter à Elasticsearch.")
                return []

            print("🔍 Connexion à Elasticsearch réussie.")
            index_name = "github_repos"
            query = {"query": {"match_all": {}}}
            response = es.search(index=index_name, body=query, size=1000)
            data = [hit["_source"] for hit in response["hits"]["hits"]]
            return data
        except Exception as e:
            print(f"⚠️ Erreur lors de la connexion à Elasticsearch: {e}")
            return []

    def check_and_train(self):
        """Vérifie le dataset et déclenche l'entraînement si nécessaire."""
        while self.running:
            print("\n🔍 Vérification du dataset dans Elasticsearch...")
            data = self.get_dataset_from_elasticsearch()
            current_data_count = len(data)

            if not data:
                print("⚠️ Dataset vide ou impossible de récupérer les données.")
            else:
                try:
                    es = Elasticsearch("http://elasticsearch:9200")
                    meta_index = "training_metadata"

                    # Vérifier s'il existe une valeur enregistrée
                    try:
                        last_record = es.get(index=meta_index, id=1)
                        stored_data_count = last_record["_source"]["data_count"]
                    except Exception:
                        stored_data_count = None

                    print(f"📊 Dernière valeur enregistrée: {stored_data_count}")
                    print(f"📈 Nombre actuel de documents: {current_data_count}")

                    # 🚀 Toujours déclencher l'entraînement si on a au moins 400 documents
                    if current_data_count >= 400:
                        print("🚀 Dataset de taille suffisante, lancement de l'entraînement !")
                        train_model()  # Appel à la fonction d'entraînement

                        # Mise à jour du nombre de documents dans Elasticsearch
                        es.index(index=meta_index, id=1, body={"data_count": current_data_count})
                    else:
                        print("✅ Dataset insuffisant, attente de la prochaine vérification.")

                except Exception as e:
                    print(f"⚠️ Erreur avec Elasticsearch: {e}")

            time.sleep(self.interval)

    def start(self):
        """Démarrer le scheduler dans un thread séparé."""
        print("⏳ Démarrage du scheduler pour surveiller Elasticsearch...")
        thread = threading.Thread(target=self.check_and_train, daemon=True)
        thread.start()


if __name__ == "__main__":
    scheduler = DatasetScheduler(interval=int(os.getenv("MODEL_TRAINING_INTERVAL")))  # Vérifie toutes les 10 minutes
    scheduler.start()

    while True:
        time.sleep(3600)  # Empêche le script de se terminer
