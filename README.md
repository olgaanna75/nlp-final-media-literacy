# Cross-Lingual RAG for Migrant Parents in Switzerland
## CAS NLP 2025/2026 — University of Bern — Final Project

A retrieval-augmented generation (RAG) system enabling migrant parents
in Switzerland to ask questions about children's media use in their own
language, with answers grounded in official Swiss federal documents
(Jugend und Medien / Jeunes et médias / Giovani e media).

## Overview

Parents can query in Albanian, Russian, Tigrinya, Spanish, or the three
national languages (German, French, Italian) and receive source-grounded
responses. Cross-lingual retrieval is handled by multilingual sentence
embeddings — no translation step required.

## Stack

- Embedding: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers), used for both local development and the GPUStack-deployed app
- Vector store: ChromaDB (persistent)
- Generation: qwen3:8b via Ollama (local development) / gpt-oss-120b via UniBE GPUStack (production)
- Evaluation: RAGAS 0.4.3 (faithfulness)
- Interface: Streamlit 1.57.0

## Notebooks

01_corpus_extraction_eda — corpus inventory and language distribution
02_chunking — structural chunking (375 chunks, 4 types)
03_embeddings_retrieval — embedding model and retrieval testing
04_rag_pipeline — RAG pipeline and qualitative spot-checks
05_evaluation — RAGAS faithfulness evaluation (GPUStack)
06_streamlit — Streamlit app launch instructions

## Vector index

`data/chroma/` is not included in this repository (generated, ~18 MB).
To recreate it, run notebooks 01-03 in order: 01 extracts and cleans the
corpus, 02 produces the chunks, 03 generates embeddings and builds the
ChromaDB collection. This must be done once before running the app.

## Setup

source venv_rag/bin/activate
pip install -r requirements.txt --break-system-packages

## Running the App

Local (requires Ollama with qwen3:8b):
ollama serve
streamlit run app/streamlit_app.py

Production (requires UniBE GPUStack):
streamlit run app/streamlit_app_gpu.py

Requires a `.env` file (not committed) with:
GPUSTACK_API_KEY=your_key_here
GPUSTACK_BASE_URL=https://gpustack.unibe.ch/v1

## Key Results

- Cross-lingual retrieval confirmed for all 8 project languages
- Tigrinya retrieval: cosine similarity up to 0.91 (highest in corpus)
- RAGAS faithfulness: 0.78 (80 instances, GPUStack)
- Tigrinya generation: confirmed working with gpt-oss-120b

Full results: report/report.md

## Author

CAS NLP 2025/2026 — University of Bern
