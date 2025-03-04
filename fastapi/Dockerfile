FROM python:3.9

WORKDIR /app

# Copier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Lancer FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]