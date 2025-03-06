FROM python:3.9

WORKDIR /app

# Copier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Définir le script Python comme point d'entrée
CMD ["python", "topic_classification_model.py"]
