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
    .option("maxOffsetsPerTrigger", "20") \
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

def load_model():
    """
    Charge et renvoie le tokenizer et le modèle. Utilisé pour chaque worker afin de charger
    les ressources locales.
    """
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

def generate_embedding(text, tokenizer, model):
    """
    Fonction pour générer des embeddings à partir d'un texte, en utilisant le tokenizer et le modèle
    préalablement chargés.
    """
    if text:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings
    return None

# Définir l'UDF pour Spark en spécifiant le type de retour
def embedding_udf(text):
    tokenizer, model = load_model()
    return generate_embedding(text, tokenizer, model)

# Appliquer l'UDF sur les colonnes 'repo_name', 'description', et 'languages'
embedding_udf_spark = udf(embedding_udf, ArrayType(FloatType()))

repos_df = repos_df.withColumn("repo_name_embedding", embedding_udf_spark(col("repo_name")))
repos_df = repos_df.withColumn("description_embedding", embedding_udf_spark(col("description")))

# Pour 'languages', convertir la liste en chaîne de caractères avant de générer l'embedding
repos_df = repos_df.withColumn("languages_embedding", embedding_udf_spark(col("languages").cast(StringType())))

def write_to_es(batch_df, batch_id):
    """
    Fonction pour écrire les résultats dans Elasticsearch.
    """
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
