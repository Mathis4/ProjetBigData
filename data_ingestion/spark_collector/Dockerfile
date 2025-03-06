FROM python:3.9

WORKDIR /app

# Installer Java (OpenJDK 17)
RUN apt-get update && apt-get install -y openjdk-17-jdk && rm -rf /var/lib/apt/lists/*

# Copier les dépendances et les installer
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY .. .

# Définir JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Ajouter l'entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
