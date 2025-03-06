#!/bin/sh
set -e

echo "Lancement du script spark_collector.py"

# Lancer le deuxième script en arrière-plan
python /app/spark_collector.py &

# Attendre que les processus terminent
wait

