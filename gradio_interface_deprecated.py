import gradio as gr
import requests

API_BASE_URL = "http://127.0.0.1:8000"

def get_github_topics(repo_url: str):
    response = requests.get(f"{API_BASE_URL}/github-topics/", params={"repo_url": repo_url})
    return f"Topics extraits : {', '.join(response.json().get('topics', []))}"

def suggest_topics(title: str, technologies: str, readme_content: str):
    payload = {"title": title, "technologies": technologies, "readme_content": readme_content}
    response = requests.post(f"{API_BASE_URL}/suggest-topics/", json=payload)
    return f"Topics suggérés : {', '.join(response.json().get('suggested_topics', []))}"

with gr.Blocks() as app:
    gr.Markdown("# Interface Gradio pour l'analyse de projets GitHub")

    with gr.Tab("Obtenir les Topics"):
        repo_url = gr.Textbox(label="URL du projet GitHub")
        output_topics = gr.Textbox(label="Topics extraits", interactive=False)
        get_topics_button = gr.Button("Extraire les Topics")
        get_topics_button.click(get_github_topics, inputs=[repo_url], outputs=[output_topics])
        
    with gr.Tab("Proposer des Topics"):
        title = gr.Textbox(label="Titre du projet")
        technologies = gr.Textbox(label="Technologies utilisées")
        readme_content = gr.Textbox(label="Contenu du README")
        suggested_topics_output = gr.Textbox(label="Topics suggérés", interactive=False)
        suggest_topics_button = gr.Button("Proposer des Topics")
        suggest_topics_button.click(
            suggest_topics, inputs=[title, technologies, readme_content], outputs=[suggested_topics_output]
        )

app.launch()
