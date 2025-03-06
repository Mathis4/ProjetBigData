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

def flatten_embedding(embedding):
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        # Aplatir une liste de listes
        return [item for sublist in embedding for item in sublist]
    elif isinstance(embedding, list):
        return embedding
    else:
        return []

def pad_embedding(embedding, length):
    return embedding + [0] * (length - len(embedding))