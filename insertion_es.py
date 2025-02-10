from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import random

# Connexion à Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Nom de l'index dans lequel on veut insérer les données
index_name = "git_repos"

# Liste des labels possibles
all_labels = ["label1", "label2", "label3"]

# Fonction pour générer des données factices avec un nombre aléatoire de labels
def generate_fake_data(n=100):
    data = []
    for _ in range(n):
        # Génération aléatoire de données pour les caractéristiques
        feature1 = random.uniform(0, 100)
        feature2 = random.uniform(0, 100)
        feature3 = random.uniform(0, 100)
        feature4 = random.uniform(0, 100)
        feature5 = random.uniform(0, 100)

        # Générer un nombre aléatoire de labels (au moins un, mais entre 1 et le nombre total de labels)
        num_labels = random.randint(1, len(all_labels))  # Tirage d'un nombre de labels aléatoire entre 1 et le total
        selected_labels = random.sample(all_labels, num_labels)  # Sélection de labels uniques

        # Créer un dictionnaire des labels sous forme de clés avec la valeur True pour chaque label présent
        label_dict = {label: (label in selected_labels) for label in all_labels}

        # Ajout des données sous forme de dictionnaire
        data.append({
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "feature4": feature4,
            "feature5": feature5,
            **label_dict,  # Ajout des labels sélectionnés
            "repo_name": f"repo_{random.randint(1, 100)}",  # Nom factice du dépôt Git
            "language": random.choice(["Python", "Java", "JavaScript", "C++", "Ruby"])  # Exemple de langage
        })
    return data

# Générer 100 données factices
fake_data = generate_fake_data(100)

# Indexer les données dans Elasticsearch
for i, doc in enumerate(fake_data):
    es.index(index=index_name, id=i, document=doc)

print(f"{len(fake_data)} documents insérés dans Elasticsearch avec succès.")
