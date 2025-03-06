#!/bin/sh
set -e

echo "Lancement du script github_collector.py"

# Lancer le premier script en arrière-plan
python /app/github_collector.py &

# Attendre que les processus terminent
wait

