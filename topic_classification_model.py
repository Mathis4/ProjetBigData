from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import json

# Étape 1 : Récupérer les données depuis Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "github_repos"
query = {"query": {"match_all": {}}}
response = es.search(index=index_name, body=query, size=1000)
data = [hit["_source"] for hit in response["hits"]["hits"]]

# Étape 2 : Préparer les données
df = pd.DataFrame(data)

# Filtrer les lignes où la colonne "topics" est vide
df = df[df["topics"].apply(lambda x: len(x) > 0)]

print(df)

# Générer la liste des labels uniques à partir des topics
unique_labels = set()
for topics in df["topics"]:
    unique_labels.update(topics)
labels = list(unique_labels)

print(unique_labels)

# Créer une colonne binaire pour chaque label
for label in labels:
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
df["description_embedding"] = df["description_embedding"].apply(flatten_embedding)
df["languages_embedding"] = df["languages_embedding"].apply(flatten_embedding)
df["repo_name_embedding"] = df["repo_name_embedding"].apply(flatten_embedding)

# Déterminer la longueur maximale des embeddings
embedding_length = max(
    df["description_embedding"].apply(len).max(),
    df["languages_embedding"].apply(len).max(),
    df["repo_name_embedding"].apply(len).max()
)

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
y = df[labels].values.astype(np.float32)

# Étape 3 : Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Créer le modèle
input_shape = X.shape[1]  # embedding_length * 3
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(labels), activation='sigmoid')
])
model.compile(optimizer='adam', loss='weighted_binary_crossentropy', metrics=['accuracy'])

# Étape 5 : Entraîner le modèle
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# Étape 6 : Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Sauvegarder les labels dans un fichier JSON
with open("labels.json", "w") as f:
    json.dump(labels, f)

# Étape 7 : Sauvegarder le modèle
model.save('my_model.keras')
print("Modèle sauvegardé avec succès sous le nom 'my_model.keras'.")

# Étape 8 : Charger le modèle sauvegardé et faire des prédictions
loaded_model = load_model('my_model.keras')

# Charger les labels à partir du fichier JSON
with open("labels.json", "r") as f:
    labels = json.load(f)

# Faire des prédictions avec le modèle chargé
predictions = loaded_model.predict(X_test)

# Conversion des prédictions en labels et affichage des topics réels
predicted_labels = []
for i in range(len(predictions)):
    predicted = []
    for j in range(len(predictions[i])):
        if predictions[i][j] > 0.5:
            predicted.append(labels[j])
    predicted_labels.append(predicted)

# Affichage des résultats avec les topics réels
for i, label_set in enumerate(predicted_labels):
    print(f"Exemple {i+1}:")
    print(f"  Topics réels: {df.iloc[i]['topics']}")
    print(f"  Prédictions: {label_set}")
