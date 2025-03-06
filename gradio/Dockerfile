FROM python:3.9

WORKDIR /app

# Copier le fichier gradio_interface.py dans le conteneur
COPY . .

# Installer Gradio
RUN pip install gradio

# Lancer Gradio
CMD ["python", "gradio_interface.py"]