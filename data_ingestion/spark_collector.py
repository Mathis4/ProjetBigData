from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import torch
from transformers import AutoTokenizer, AutoModel

# Initialisation de SparkSession
spark = SparkSession.builder \
    .appName("GitHubReposKafka") \
    .getOrCreate()

# Définition du schéma pour les données JSON
schema = StructType([
    StructField("repo_name", StringType(), True),
    StructField("full_name", StringType(), True),
    StructField("html_url", StringType(), True),
    StructField("description", StringType(), True),
    StructField("created_at", StringType(), True),
    StructField("topics", ArrayType(StringType()), True),
    StructField("languages", ArrayType(StringType()), True)
])

# Lecture du flux Kafka
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "github_repos") \
    .option("startingOffsets", "earliest") \
    .load()

# Décodage de la valeur JSON du message Kafka
kafka_stream_decoded = kafka_stream.selectExpr("CAST(value AS STRING) as json_value")

# Extraction des champs JSON en utilisant le schéma défini
repos_df = kafka_stream_decoded.select(from_json(col("json_value"), schema).alias("repo"))

# Sélectionner les colonnes spécifiques de la structure JSON
repos_df = repos_df.select(
    "repo.repo_name",
    "repo.full_name",
    "repo.html_url",
    "repo.description",
    "repo.created_at",
    "repo.topics",
    "repo.languages"
)

# Variables globales pour le chargement paresseux du tokenizer et du modèle sur chaque worker
global_tokenizer = None
global_model = None

def generate_embedding_lazy(text):
    """
    Fonction UDF pour générer des embeddings à partir d'un texte.
    Le modèle et le tokenizer sont chargés paresseusement sur chaque worker.
    """
    global global_tokenizer, global_model
    if global_tokenizer is None or global_model is None:
        # Charger le tokenizer et le modèle sur le worker (uniquement lors du premier appel)
        global_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        global_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    if text:
        # Affichage de progression (ce print apparaîtra dans les logs de l'executor)
        inputs = global_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        with torch.no_grad():
            embeddings = global_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings
    return None

# Définir l'UDF pour Spark en spécifiant le type de retour
embedding_udf = udf(generate_embedding_lazy, ArrayType(FloatType()))

# Ajouter des colonnes d'embeddings pour 'repo_name', 'description' et 'languages'
repos_df = repos_df.withColumn("repo_name_embedding", embedding_udf(col("repo_name")))
repos_df = repos_df.withColumn("description_embedding", embedding_udf(col("description")))
# Pour 'languages', convertir la liste en chaîne de caractères avant de générer l'embedding
repos_df = repos_df.withColumn("languages_embedding", embedding_udf(col("languages").cast(StringType())))

# # Afficher le contenu du DataFrame en mode streaming dans la console
# query = repos_df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("numRows", 20) \
#     .start()

def write_to_es(batch_df, batch_id):
    print(f"Traitement du lot {batch_id} avec {batch_df.count()} lignes.")
    batch_df.write \
        .format("org.elasticsearch.spark.sql") \
        .option("es.nodes", "elasticsearch") \
        .option("es.port", "9200") \
        .option("es.resource", "github_repos") \
        .option("es.index.auto.create", "true") \
        .mode("append") \
        .save()


# Remplacer l'écriture sur console par foreachBatch vers Elasticsearch
es_query = repos_df.writeStream \
    .foreachBatch(write_to_es) \
    .option("checkpointLocation", "/tmp/spark_checkpoint") \
    .trigger(processingTime="1 minute") \
    .start()

# Attendre indéfiniment l'exécution du stream
es_query.awaitTermination()
