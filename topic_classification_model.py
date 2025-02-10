from elasticsearch import Elasticsearch
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Étape 1 : Récupérer les données depuis Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "git_repos"
query = {"query": {"match_all": {}}}
response = es.search(index=index_name, body=query, size=1000)
data = [hit["_source"] for hit in response["hits"]["hits"]]

# Étape 2 : Préparer les données
df = pd.DataFrame(data)
features = ["feature1", "feature2", "feature3", "feature4", "feature5"]
labels = ["label1", "label2", "label3"]  # Remplacez par vos noms de labels
X = df[features]
y = df[labels]

# Étape 3 : Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Créer le modèle
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(labels), activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Étape 5 : Entraîner le modèle
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Étape 6 : Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Étape 7 : Faire des prédictions
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)
print(predicted_labels)