import json
import requests
from kafka import KafkaProducer
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from langdetect import detect
import time

# Variables de configuration
GITHUB_TOKEN = "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"  # Remplacez par votre token GitHub
KAFKA_TOPIC = "github_repos"
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"

# Initialisation du producteur Kafka
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


# Fonction pour rechercher des dépôts GitHub
def search_github_repos(since_date=None):
    if since_date:
        query = f"created:>{since_date}"
    else:
        query = "created:>2023-01-01"

    url = f"https://api.github.com/search/repositories?q={query}&sort=created&order=desc&per_page=100"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    repos = []
    while url:
        response = requests.get(url, headers=headers)

        if response.status_code == 403:
            # Vérifier si c'est une erreur de quota
            remaining_quota = response.headers.get('X-RateLimit-Remaining')
            reset_time = response.headers.get('X-RateLimit-Reset')
            if remaining_quota == "0":
                reset_timestamp = int(reset_time)
                reset_time = datetime.fromtimestamp(reset_timestamp)
                print(f"Quota atteint. Réessayer après {reset_time}")
                sleep_time = (reset_time - datetime.now()).total_seconds()
                time.sleep(sleep_time)
                continue  # Réessayer après le quota est réinitialisé

        if response.status_code == 200:
            data = response.json()
            repos.extend(data.get("items", []))
            url = response.links.get('next', {}).get('url', None)
        else:
            print(f"Erreur lors de la récupération des dépôts : {response.status_code}")
            break

    return repos


# Fonction pour récupérer les langages de programmation d'un dépôt
def get_languages(owner, repo_name):
    languages_url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(languages_url, headers=headers)
    if response.status_code == 200:
        return list(response.json().keys())  # Liste des langages de programmation utilisés
    else:
        print(f"Erreur lors de la récupération des langages pour {owner}/{repo_name}")
        return []


# Fonction pour envoyer les dépôts à Kafka
def send_to_kafka(repos):
    for repo in repos:
        try:
            description = repo.get("description", "")
            if description:
                try:
                    # Filtrer les descriptions en anglais
                    detected_lang = detect(description)
                    if detected_lang != 'en':
                        continue  # Ignorer si la langue n'est pas l'anglais
                except:
                    continue  # Ignorer si la détection échoue

            owner, repo_name = repo["full_name"].split("/")  # Extraire le propriétaire et le nom du dépôt
            languages = get_languages(owner, repo_name)

            repo_data = {
                "repo_name": repo["name"],
                "full_name": repo["full_name"],
                "html_url": repo["html_url"],
                "description": description,
                "created_at": repo.get("created_at", ""),
                "topics": repo.get("topics", []),
                "languages": languages  # Ajouter les langages de programmation
            }
            producer.send(KAFKA_TOPIC, value=repo_data)
            print(f"Dépôt envoyé à Kafka : {repo['name']}")
        except Exception as e:
            print(f"Erreur lors de l'envoi du dépôt {repo['name']} à Kafka: {str(e)}")


# Fonction principale pour récupérer et envoyer les dépôts
def collect_and_send(since_date=None):
    if since_date is None:
        # Si since_date n'est pas fourni, on récupère les dépôts des dernières heures
        since_date = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')  # Format date-heure précis

    print(f"Recherche des dépôts GitHub créés depuis {since_date}...")
    repos = search_github_repos(since_date)

    if repos:
        send_to_kafka(repos)
    else:
        print(f"Aucun dépôt trouvé.")


# Planification de l'exécution automatique avec APScheduler
def start_scheduler():
    scheduler = BackgroundScheduler()
    # Planifie la collecte de données toutes les heures
    scheduler.add_job(collect_and_send, 'interval', hours=1)
    scheduler.start()


if __name__ == "__main__":
    try:
        print("Récupération des dépôts du dernier mois pour l'exemple (ajustable).")
        # Récupérer les dépôts du dernier mois
        since_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        collect_and_send(since_date=since_date)

        # Démarrer le scheduler pour exécuter la tâche de collecte de données automatiquement toutes les heures
        start_scheduler()
        print("Scheduler démarré. Récupération des dépôts toutes les heures.")

        # L'application continue de s'exécuter
        while True:
            pass  # Laisser le programme en cours d'exécution
    except KeyboardInterrupt:
        print("Arrêt du script.")
    finally:
        producer.close()