FROM python:3.9

WORKDIR /app

# Copier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

CMD ["tail", "-f", "/dev/null"]
