# 🔍 GitHub Topic Suggester

## 🚀 À propos
Cette application permet de prédire des **topics** pour un dépôt GitHub en se basant sur ses caractéristiques.  
Pour ce faire, elle entraîne des modèles de classification multi-labels, le modèle ayant les meilleures métriques
sera chargé pour la prédiction de l'utilisateur.

## 🛠 Fonctionnalités
- 🔹 Visualisation des modèles et de leurs métriques via **MLflow** → [http://localhost:5000](http://localhost:5000)
- 🔹 Visualisation des données via **Kibana** → [http://localhost:5601/app/discover#/](http://localhost:5601/app/discover#/)
- 🔹 Interface interactive pour explorer les recommandations via **Gradio** → [http://localhost:7860](http://localhost:7860)


## 📦 Installation

### 1️⃣ Prérequis  
Avant d'exécuter l'application, assurez-vous d'avoir :  
- 🐳 **Docker** installé → [Instructions d'installation](https://docs.docker.com/get-docker/)  
- 🔑 Un **token API GitHub** à renseigner dans le fichier `.env`  

### 2️⃣ Lancer avec Docker
Clonez le dépôt, mettez-vous à la racine du projet et démarrez les services :
```
docker-compose build
docker-compose up -d
```
Une fois les commandes faites, il vous reste plus qu'à attendre qu'il y ait assez de données pour que l'entrainement 
d'un modèle se fasse, vous pourrez ensuite vous rendre sur l'interface gradio pour faire une prédiction.