# streamlit_app.py — RAG interface for migrant parents
# Jugend und Medien corpus — cross-lingual QA system
# CAS NLP 2025 — University of Bern

import streamlit as st
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# configuration
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "jugend_medien"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
TOP_K = 5

@st.cache_resource
def load_pipeline():
    # load embedding model and ChromaDB collection once
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection

def retrieve(query, model, collection, top_k=TOP_K):
    # embed query and retrieve top-k chunks
    embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]

def generate(query, contexts):
    # build prompt and call Ollama
    context_block = "\n\n".join(contexts)
    prompt = (
        f"Answer the following question based only on the provided sources.\n"
        f"Use simple, clear language suitable for a parent.\n"
        f"Reply in the same language as the question.\n\n"
        f"Sources:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"].strip()

def main():
    st.set_page_config(
        page_title="Media Literacy for Parents",
        page_icon="💬",
        layout="centered"
    )

    st.title("💬 Media Literacy for Parents")

    st.caption(
    "Media literacy / Compétences médiatiques / Medienkompetenz / "
    "Competenze mediali / Медиаграмотность / "
    "Alfabetización mediática / Kompetenca mediatike / ሚድያ ብቕዓት"
    )

    st.markdown(
    "Ask a question about children's media use in your own language. "
    "Answers are grounded in official Swiss Jugend und Medien materials. "
    "Supported languages: German · French · Italian · English · "
    "Russian · Albanian · Tigrinya · Spanish."
    )

    model, collection = load_pipeline()

    query = st.text_input(
        "Your question / Votre question / Ihre Frage / La sua domanda / "
        "Вопрос / Su pregunta / Pyetja juaj / ሕቶኹም",
        placeholder="My child is 3 years old. How much screen time per day is okay?"
    )

    if st.button(
        "Ask / Envoyer / Absenden / Invia / Спросить / Preguntar / Pyesni / ሕተቱ"
    ) and query.strip():
        with st.spinner("Searching sources and generating answer..."):
            docs, metas = retrieve(query, model, collection)
            answer = generate(query, docs)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Sources used"):
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                st.markdown(f"**[{i+1}] {meta.get('source', 'unknown')}**")
                st.caption(doc[:200] + "...")

if __name__ == "__main__":
    main()