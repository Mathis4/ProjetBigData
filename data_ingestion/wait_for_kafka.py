import time
from kafka import KafkaClient
from kafka.errors import NoBrokersAvailable

# Variables de configuration
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"  # Change le nom du service kafka si nécessaire

def wait_for_kafka(bootstrap_servers, retries=10, delay=5):
    for _ in range(retries):
        try:
            client = KafkaClient(bootstrap_servers=bootstrap_servers)
            client.cluster  # Essaye de récupérer des informations de cluster pour vérifier la connexion
            print("Kafka est prêt!")
            return True
        except NoBrokersAvailable:
            print(f"Kafka non disponible, tentative dans {delay} secondes...")
            time.sleep(delay)
    print("Kafka n'est toujours pas disponible après plusieurs tentatives.")
    return False

if __name__ == "__main__":
    if wait_for_kafka(KAFKA_BOOTSTRAP_SERVERS):
        print("Kafka prêt, démarrage de l'application Python.")
    else:
        print("Échec de la connexion à Kafka.")
        exit(1)