# streamlit_app_gpu.py — RAG interface for migrant parents
# Jugend und Medien corpus — cross-lingual QA system
# CAS NLP 2025 — University of Bern
# Production version: UniBE GPUStack (gpt-oss-120b)

# imports
import os
import streamlit as st
import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# load environment variables from .env
load_dotenv()

# configuration
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "jugend_medien"
GPUSTACK_URL = "https://gpustack.unibe.ch/v1/chat/completions"
GPUSTACK_MODEL = "gpt-oss-120b"
GPUSTACK_API_KEY = os.environ.get("GPUSTACK_API_KEY", "")
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
    # load system prompt and call GPUStack
    context_block = "\n\n".join(contexts)
    system_prompt = open("prompts/system_prompt.txt", encoding="utf-8").read()
    user_message = (
        f"Question: {query}\n\n"
        f"Source excerpts:\n{context_block}\n\n"
        f"Please answer the question based on the source excerpts above."
    )
    response = requests.post(
        GPUSTACK_URL,
        headers={
            "Authorization": f"Bearer {GPUSTACK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GPUSTACK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 3000,
            "stream": False
        }
    )
    return response.json()["choices"][0]["message"]["content"].strip()

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