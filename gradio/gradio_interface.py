import gradio as gr
import requests
import base64

API_BASE_URL = "http://fastapi:8000"

# Charger l'image "github.png" depuis le répertoire du projet et la convertir en base64
with open("github.png", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
img_data_uri = f"data:image/png;base64,{img_base64}"


def get_github_topics(repo_url: str):
    try:
        response = requests.get(f"{API_BASE_URL}/github-topics/", params={"repo_url": repo_url})
        if response.status_code == 200:
            topics = response.json().get("topics", [])
            if topics:
                badges = " ".join([f'<span class="badge">{topic}</span>' for topic in topics])
                return badges
            else:
                return '<span style="color: red;">❌ Aucun topic trouvé.</span>'
        return f'<span style="color: red;">🚨 Erreur {response.status_code} : {response.text}</span>'
    except requests.ConnectionError:
        return '<span style="color: red;">⚠️ Impossible de se connecter au backend FastAPI.</span>'


def suggest_topics(title: str, technologies: str, readme_content: str):
    payload = {
        "title": title,
        "technologies": technologies,
        "readme_content": readme_content
    }
    try:
        response = requests.post(f"{API_BASE_URL}/suggest-topics-details/", json=payload)
        if response.status_code == 200:
            topics = response.json().get("suggested_topics", [])
            if topics:
                badges = " ".join([f'<span class="badge">{topic}</span>' for topic in topics])
                return badges
            else:
                return '<span style="color: red;">❌ Aucun topic suggéré.</span>'
        return f'<span style="color: red;">🚨 Erreur {response.status_code} : {response.text}</span>'
    except requests.ConnectionError:
        return '<span style="color: red;">⚠️ Impossible de se connecter au backend FastAPI.</span>'


# CSS personnalisé pour un design épuré, moderne et pour styliser les badges
custom_css = """
body {
    font-family: 'Roboto', sans-serif;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    color: #333;
}
.container {
    max-width: 800px;
    margin: 2rem auto;
    padding: 2rem;
    background-color: #f9f9f9;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.header, .footer {
    text-align: center;
    padding: 1rem;
}
.header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: #007BFF;
}
.header p {
    font-size: 1.2rem;
    color: #777;
}
.section {
    margin: 1rem 0;
    padding: 1.5rem;
    background-color: #fff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}
.button {
    background-color: #007BFF;
    border: none;
    color: #fff;
    padding: 0.8rem 1.2rem;
    font-size: 1rem;
    border-radius: 5px;
    cursor: pointer;
}
.button:hover {
    background-color: #0056b3;
}
/* Style pour les badges */
.badge {
    display: inline-block;
    background-color: #007BFF;
    color: #fff;
    padding: 5px 10px;
    border-radius: 15px;
    margin: 2px;
    font-size: 0.9rem;
}
"""

with gr.Blocks(css=custom_css) as app:
    # Header avec l'image locale et explication de l'application
    gr.HTML(f"""
    <div class="container">
      <div class="header">
        <img src="{img_data_uri}" alt="Bannière" style="width:100%; border-radius:10px;">
        <h1>🚀 Analyse de Projets GitHub</h1>
        <p>Extraire et suggérer des topics grâce à l'intelligence artificielle</p>
      </div>
    """)

    gr.Markdown("""
    <div style="text-align: center; font-size:1.1rem; line-height:1.6; margin-bottom: 1.5rem;">
      <strong>À propos de l'application :</strong><br>
      Cette application vous permet de <em>détecter automatiquement</em> les topics d'un dépôt GitHub en utilisant l'API GitHub,<br>
      puis de <em>proposer intelligemment</em> des topics en se basant sur les caractéristiques de votre projet.<br>
      Utilisez les onglets ci-dessous pour explorer chacune des fonctionnalités.
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🔍 Extraire des Topics"):
            with gr.Row():
                repo_url_input = gr.Textbox(label="🔗 URL du projet GitHub", placeholder="Ex : octocat/Hello-World")
                get_topics_button = gr.Button("Extraire", elem_classes=["button"])
            output_topics = gr.HTML(label="📌 Topics extraits")
            get_topics_button.click(get_github_topics, inputs=[repo_url_input], outputs=[output_topics])

        with gr.Tab("💡 Proposer des Topics"):
            with gr.Row():
                title_input = gr.Textbox(label="📌 Titre du projet", placeholder="Ex : Analyse de données")
            with gr.Row():
                technologies_input = gr.Textbox(label="🛠 Technologies utilisées", placeholder="Ex : Python, TensorFlow")
            with gr.Row():
                readme_content_input = gr.Textbox(label="📖 Contenu du README", placeholder="Description du projet...")
            suggest_topics_button = gr.Button("Générer des Topics", elem_classes=["button"])
            suggested_topics_output = gr.HTML(label="📌 Topics suggérés")
            suggest_topics_button.click(
                suggest_topics,
                inputs=[title_input, technologies_input, readme_content_input],
                outputs=[suggested_topics_output]
            )

    gr.HTML("""
      <div class="footer">
        <p>Powered by FastAPI & Gradio</p>
      </div>
    </div>
    """)

app.launch(server_name="0.0.0.0", server_port=7860)