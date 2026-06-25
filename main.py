"""
Resume RAG Backend
------------------
Learn: embeddings · chunking · vector search
Stack: FastAPI · OpenAI embeddings · Chroma · Groq LLM
"""

import os, json, re, textwrap
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from groq import Groq
from pydantic import BaseModel
import PyPDF2
import io

from sentence_transformers import SentenceTransformer

load_dotenv()
# ── clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
groq_client   = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Chroma (in-memory for the demo) ──────────────────────────────────────────
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

COLLECTION_NAME = "resume"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads once, ~90MB
# wipe & recreate so each upload starts fresh
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

app = FastAPI(title="Resume RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """
    CHUNKING 101
    ─────────────
    We split the raw text into overlapping windows.

    • chunk_size  – target token count per chunk (≈ words here)
    • overlap     – tokens shared between consecutive chunks so context
                    doesn't get cut at a boundary

    Why overlap?  A sentence like "Led a team of 8 engineers" might start
    at the very end of chunk N. Without overlap, chunk N+1 misses the
    beginning and the meaning is lost.
    """
    words  = text.split()
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "id":    f"chunk_{idx}",
            "text":  chunk,
            "start": start,
            "end":   end,
        })
        idx   += 1
        start += chunk_size - overlap   # slide window with overlap

    return chunks




def embed(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()
    """
    EMBEDDINGS 101
    ──────────────
    An embedding converts text → a list of floats (a vector).
    Semantically similar text ends up close together in vector space.

    Example:
      "Python developer" ≈ "software engineer" (high cosine similarity)
      "Python developer" ≠ "pizza chef"        (low cosine similarity)

    We use OpenAI's text-embedding-3-small (1 536 dimensions).
    Each dimension captures some latent semantic feature of the text.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """
    VECTOR SEARCH 101
    ─────────────────
    1. Embed the user's question → query_vector
    2. Compute cosine similarity between query_vector and every stored chunk
    3. Return the top-k closest chunks

    Cosine similarity = dot(A, B) / (|A| * |B|)
    Range: -1 (opposite) → 0 (unrelated) → 1 (identical)

    Chroma does this with an HNSW index (Hierarchical Navigable Small World)
    — an ANN (approximate nearest neighbour) graph that scales to millions
    of vectors without brute-forcing every pair.
    """
    [query_vec] = embed([query])
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    hits = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        hits.append({
            "text":       doc,
            "similarity": round(1 - dist, 4),   # cosine distance → similarity
            "chunk_id":   meta.get("chunk_id"),
        })
    return hits


# ── routes ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Parse PDF → chunk → embed → store in Chroma."""
    global collection

    raw = await file.read()

    # extract
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(raw)
    else:
        text = raw.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "Could not extract text from the file.")

    # chunk
    chunks = chunk_text(text, chunk_size=400, overlap=80)

    # embed  (batch to stay within rate limits)
    batch_size = 20
    all_ids, all_docs, all_metas, all_embeddings = [], [], [], []

    for i in range(0, len(chunks), batch_size):
        batch   = chunks[i : i + batch_size]
        texts   = [c["text"] for c in batch]
        vectors = embed(texts)
        for c, v in zip(batch, vectors):
            all_ids.append(c["id"])
            all_docs.append(c["text"])
            all_metas.append({"chunk_id": c["id"], "start": c["start"], "end": c["end"]})
            all_embeddings.append(v)

    # wipe old data, insert fresh
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas,
        embeddings=all_embeddings,
    )

    return {
        "message":       "Resume indexed successfully!",
        "total_chunks":  len(chunks),
        "total_vectors": collection.count(),
        "sample_chunks": [c["text"][:120] + "…" for c in chunks[:3]],
    }


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


@app.post("/chat")
async def chat(req: ChatRequest):
    """Retrieve relevant chunks → feed to Groq → return answer + debug info."""
    if collection.count() == 0:
        raise HTTPException(400, "No resume indexed yet. Upload a PDF first.")

    # retrieve
    hits = vector_search(req.question, top_k=req.top_k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)

    # augment + generate
    system_prompt = textwrap.dedent("""
        You are an expert career assistant. Answer questions about the candidate
        strictly using the resume excerpts provided. Be specific, cite skills,
        companies, and dates when relevant. If the answer isn't in the context,
        say so honestly.
    """).strip()

    user_prompt = f"""Resume excerpts (most relevant first):
{context}

Question: {req.question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "answer":          answer,
        "retrieved_chunks": hits,
        "model":           "llama-3.3-70b-versatile (Groq)",
        "embedding_model": "text-embedding-3-small (OpenAI)",
    }


@app.get("/debug/chunks")
async def debug_chunks():
    """Return all stored chunks for inspection."""
    if collection.count() == 0:
        return {"chunks": []}
    result = collection.get(include=["documents", "metadatas"])
    return {
        "total": collection.count(),
        "chunks": [
            {"id": cid, "text": doc[:200], "meta": meta}
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "indexed_chunks": collection.count()}
