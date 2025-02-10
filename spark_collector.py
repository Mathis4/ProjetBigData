import socket
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Création de la session Spark
spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("GitHubCollector") \
    .config("spark.metrics.enabled", "false") \
    .getOrCreate()

# Définissez le schéma des données envoyées sur Kafka
schema = StructType([
    StructField("repo_name", StringType(), True),
    StructField("full_name", StringType(), True),
    StructField("html_url", StringType(), True),
    StructField("description", StringType(), True),
    StructField("created_at", StringType(), True),
    StructField("topics", ArrayType(StringType()), True)
])

# Configuration pour consommer les messages Kafka
kafka_bootstrap_servers = "kafka:9092"  # Référence au conteneur Kafka
kafka_topic = "github_repos"

# Lire le stream de Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .load()

# Afficher le schéma pour vérifier les données Kafka
df.printSchema()

# Convertir les valeurs en JSON structuré
df = df.selectExpr("CAST(value AS STRING)")

# Vérifier le contenu des données après conversion
df.show(5, False)

# Appliquer le schéma et extraire les colonnes
df_parsed = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Afficher les résultats après parsing
df_parsed.show(5, False)

# Afficher les résultats dans la console
query = df_parsed \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

print("Streaming started...")  # Message de contrôle

# Attendre la fin du stream
query.awaitTermination()