from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from transformers import AutoTokenizer, AutoModel
import torch

# Charger un modèle d'embedding pré-entraîné
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Fonction pour générer les embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

# Définir l'UDF pour Spark
generate_embedding_udf = udf(lambda text: generate_embedding(text), ArrayType(FloatType()))

# Ajouter des colonnes d'embeddings pour 'repo_name', 'description' et 'languages'
repos_df = repos_df.withColumn("repo_name_embedding", generate_embedding_udf(col("repo_name")))
repos_df = repos_df.withColumn("description_embedding", generate_embedding_udf(col("description")))

# Pour les 'languages', on peut les joindre en une seule chaîne avant de générer l'embedding
repos_df = repos_df.withColumn("languages_embedding", generate_embedding_udf(col("languages").cast(StringType())))

# Affichage des résultats (pour les tests en mode batch)
repos_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start() \
    .awaitTermination()