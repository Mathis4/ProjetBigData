import json
import requests
from kafka import KafkaProducer
from datetime import datetime
from langdetect import detect, LangDetectException

# Variables de configuration
GITHUB_TOKEN = "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"  # Remplacez par votre token GitHub
KAFKA_TOPIC = "github_repos"
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"

# Langues à filtrer (français et anglais)
ALLOWED_LANGUAGES = ["en"]  # 'en' pour anglais, 'fr' pour français

# Initialisation du producteur Kafka
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Fonction pour rechercher des dépôts GitHub créés dans un mois spécifique
def search_github_repos(month, year):
    query = f"created:{year}-{month:02d}-01..{year}-{month:02d}-30"
    url = f"https://api.github.com/search/repositories?q={query}&sort=created&order=desc&per_page=100"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    repos = []
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            repos.extend(data.get("items", []))
            # Vérifiez s'il existe une page suivante
            url = response.links.get('next', {}).get('url', None)
        else:
            print(f"Erreur lors de la récupération des dépôts : {response.status_code}")
            return []

    return repos

# Fonction pour détecter la langue du texte
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

# Fonction pour filtrer les dépôts selon la langue naturelle (français ou anglais)
def filter_repos_by_language(repos):
    filtered_repos = []
    for repo in repos:
        description = repo.get("description", "")
        if description:  # Vérifier si la description n'est pas vide
            lang = detect_language(description)
            if lang in ALLOWED_LANGUAGES:
                filtered_repos.append(repo)
    return filtered_repos

# Fonction pour envoyer les noms des dépôts à Kafka
def send_to_kafka(repos):
    for repo in repos:
        try:
            repo_data = {
                "repo_name": repo["name"],
                "full_name": repo["full_name"],
                "html_url": repo["html_url"],
                "description": repo.get("description", ""),
                "created_at": repo.get("created_at", ""),
                "topics": repo.get("topics", []),
            }
            producer.send(KAFKA_TOPIC, value=repo_data)
            print(f"Dépôt envoyé à Kafka : {repo['name']}")
        except Exception as e:
            print(f"Erreur lors de l'envoi du dépôt {repo['name']} à Kafka: {str(e)}")

# Fonction principale pour récupérer et envoyer les dépôts
def collect_and_send(year):
    print(f"Recherche des dépôts GitHub créés en {year}...")

    # Parcours de chaque mois de l'année
    for month in range(1, 2):
        print(f"Recherche des dépôts GitHub créés en {month}/{year}...")
        repos = search_github_repos(month, year)
        
        if repos:
            # Filtrer les dépôts par langue naturelle
            filtered_repos = filter_repos_by_language(repos)
            if filtered_repos:
                send_to_kafka(filtered_repos)
            else:
                print(f"Aucun dépôt trouvé en anglais ou français pour {month}/{year}.")
        else:
            print(f"Aucun dépôt trouvé pour {month}/{year}.")


if __name__ == "__main__":
    try:
        # Exemple : récupérer les repos créés en 2024
        collect_and_send(2024)
    except KeyboardInterrupt:
        print("Arrêt du script.")
    finally:
        producer.close()
