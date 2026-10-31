# Project Documentation SLM Chatbot

> A fully local, private Small Language Model (SLM) chatbot specialized for project documentation with document upload and RAG. No API keys required. Runs entirely on your machine.

## What it does

Upload project documentation (PDF or TXT) and ask questions about it. The chatbot retrieves relevant sections from your project documents and uses a local **Small Language Model (SLM)** such as **Gemma 3n E2B** (via Ollama) to answer accurately — all locally, all private.

Unlike generic Large Language Models (LLMs), this chatbot operates with a dedicated SLM persona specifically tailored to answer questions regarding your project's documentation.

---

## Stack

| Layer | Tool |
|---|---|
| Model (SLM) | Gemma 3n E2B · Ollama (Small Language Model) |
| Backend | Python · FastAPI · LangChain |
| Vector DB | ChromaDB / BM25 Store |
| Embeddings | sentence-transformers |
| Frontend | HTML · CSS · JS |
| Deploy | Docker · docker-compose |

---

## Setup

> Full walkthrough → **[SETUP.txt](./SETUP.txt)**

```bash
# 1. Pull the model
ollama pull gemma3n:e2b

# 2. Create environment
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install and run
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000**

**Docker alternative:**
```bash
docker-compose up --build
```

---

## Project structure

```
rag-chatbot/
├── backend/
│   ├── main.py            API server
│   ├── rag.py             RAG pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html         Chat UI
├── chromadb_data/         Auto-created · stores embeddings
├── uploads/               Auto-created · stores your files
├── docker-compose.yml
├── README.md
└── SETUP.txt              Full setup guide
```

---

## Common errors

| Error | Fix |
|---|---|
| Connection refused :11434 | `ollama serve` |
| model not found | `ollama pull gemma3n:e2b` |
| Activate.ps1 blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Slow responses | Normal on CPU — just wait |

---

*Built for AI Advanced Skill Development Lab*
