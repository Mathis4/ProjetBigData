from elasticsearch import Elasticsearch

# Connexion au cluster Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"])  # Remplacez l'URL si nécessaire

# Vérification de la connexion
if es.ping():
    print("Connexion réussie à Elasticsearch")
else:
    print("Connexion échouée à Elasticsearch")
    exit()

# Étape 1 : Création d'un index
index_name = "github_repos"

# Définir le mappage pour les champs de l'index
mapping = {
    "mappings": {
        "properties": {
            "repo_name": {"type": "text"},
            "full_name": {"type": "text"},
            "html_url": {"type": "text"},
            "description": {"type": "text"},
            "created_at": {"type": "date"}
        }
    }
}

# Créer l'index si nécessaire
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' créé avec succès")
else:
    print(f"L'index '{index_name}' existe déjà")

# Étape 2 : Ajouter des documents
documents = [
    {
        "repo_name": "example-repo-1",
        "full_name": "user/example-repo-1",
        "html_url": "https://github.com/user/example-repo-1",
        "description": "Un dépôt d'exemple",
        "created_at": "2025-01-01T12:00:00Z"
    },
    {
        "repo_name": "example-repo-2",
        "full_name": "user/example-repo-2",
        "html_url": "https://github.com/user/example-repo-2",
        "description": "Un deuxième dépôt d'exemple",
        "created_at": "2025-01-05T08:30:00Z"
    }
]

# Ajouter chaque document à l'index
for i, doc in enumerate(documents, start=1):
    es.index(index=index_name, id=i, document=doc)
    print(f"Document {i} ajouté à l'index '{index_name}'")

# Étape 3 : Rechercher dans l'index
query = {
    "query": {
        "match": {
            "repo_name": "example"
        }
    }
}

response = es.search(index=index_name, body=query)
print("Résultats de la recherche :")
for hit in response["hits"]["hits"]:
    print(hit["_source"])
