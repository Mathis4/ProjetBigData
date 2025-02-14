import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.title("Analyse et Suggestion de Topics pour Projets GitHub")

tab1, tab2 = st.tabs(["Obtenir les Topics", "Proposer des Topics"])

with tab1:
    st.header("Obtenir les Topics")
    repo_url = st.text_input("URL du projet GitHub")
    if st.button("Extraire les Topics"):
        response = requests.get(f"{API_BASE_URL}/github-topics/", params={"repo_url": repo_url})
        topics = response.json().get("topics", [])
        st.success(f"Topics extraits : {', '.join(topics)}")

with tab2:
    st.header("Proposer des Topics")
    title = st.text_input("Titre du projet")
    technologies = st.text_input("Technologies utilisées")
    readme_content = st.text_area("Contenu du README")
    if st.button("Proposer des Topics"):
        payload = {"title": title, "technologies": technologies, "readme_content": readme_content}
        response = requests.post(f"{API_BASE_URL}/suggest-topics/", json=payload)
        suggested_topics = response.json().get("suggested_topics", [])
        st.success(f"Topics suggérés : {', '.join(suggested_topics)}")
