# 🧠 Resume RAG — Chat with your CV

> **Learn:** embeddings · chunking · vector search  
> **Build:** a full RAG pipeline powered by OpenAI + ChromaDB + Groq

---

## Stack

| Layer | Tool |
|---|---|
| API server | FastAPI |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | ChromaDB (HNSW index, cosine similarity) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Frontend | Vanilla HTML/JS (single file) |

---

## How it works (RAG in 3 steps)

```
PDF ──► CHUNK ──► EMBED ──► ChromaDB
                               │
Question ──► EMBED             │
                 └─► SEARCH ──►┘
                        │
                    top-k chunks
                        │
                     GROQ LLM ──► Answer
```

### 1. Chunking (`chunk_text`)
Raw text is split into **400-word windows with 80-word overlap**.  
Overlap prevents context loss at chunk boundaries.

### 2. Embeddings (`embed`)
Each chunk is converted to a **1 536-dimension float vector** via  
`text-embedding-3-small`. Similar meaning → nearby vectors.

### 3. Vector Search (`vector_search`)
The user's question is embedded, then ChromaDB finds the **top-5 closest chunks**  
using cosine similarity via an HNSW (Hierarchical Navigable Small World) index.

---

## Quick Start

### Prerequisites
```bash
pip install -r backend/requirements.txt
```

### Set API keys
```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

### Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Open the frontend
```bash
open frontend/index.html
# or serve it:
python -m http.server 3000 --directory frontend
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDF → chunk → embed → store |
| `POST` | `/chat` | Ask a question, get RAG answer |
| `GET` | `/debug/chunks` | Inspect all stored chunks |
| `GET` | `/health` | Health check |

### Example: Chat request
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What programming languages do I know?", "top_k": 5}'
```

### Example: Response
```json
{
  "answer": "Based on your resume, you are proficient in Python, Go, and TypeScript...",
  "retrieved_chunks": [
    { "text": "...", "similarity": 0.89, "chunk_id": "chunk_2" }
  ],
  "model": "llama-3.3-70b-versatile (Groq)",
  "embedding_model": "text-embedding-3-small (OpenAI)"
}
```

---

## Key concepts in the code

```python
# chunking — see main.py::chunk_text()
words  = text.split()
chunks = []
start  = 0
while start < len(words):
    end   = min(start + chunk_size, len(words))
    chunk = " ".join(words[start:end])
    chunks.append(chunk)
    start += chunk_size - overlap   # ← overlap is the key

# embedding — see main.py::embed()
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)

# vector search — see main.py::vector_search()
collection.query(
    query_embeddings=[query_vec],
    n_results=top_k,
    include=["documents", "distances"],
)
```

---

## Project structure

```
resume-rag/
├── backend/
│   ├── main.py           # FastAPI + RAG pipeline
│   └── requirements.txt
└── frontend/
    └── index.html        # Chat UI (single file, no build step)
```
