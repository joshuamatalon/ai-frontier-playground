# vec_memory.py
import os, uuid
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# --- env ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "cca-memories")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM        = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small = 1536

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing.")

# --- clients ---
oa = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- ensure index exists (serverless) ---
try:
    index_names = set(pc.list_indexes().names())
except Exception:
    index_names = set(i["name"] for i in pc.list_indexes())

if INDEX_NAME not in index_names:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

index = pc.Index(INDEX_NAME)

# --- embeddings ---
def _embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# --- public API (unchanged signature) ---
def upsert_note(text: str, meta: Dict[str, Any] | None = None) -> str:
    _id = str(uuid.uuid4())
    vec = _embed([text])[0]
    index.upsert(vectors=[{"id": _id, "values": vec, "metadata": {"text": text, **(meta or {})}}])
    return _id

def upsert_many(chunks: List[str], meta: Dict[str, Any]) -> List[str]:
    if not chunks:
        return []
    ids  = [str(uuid.uuid4()) for _ in chunks]
    vecs = _embed(chunks)
    index.upsert(vectors=[
        {"id": i, "values": v, "metadata": {"text": t, **meta}}
        for i, v, t in zip(ids, vecs, chunks)
    ])
    return ids

def search(query: str, k: int = 5) -> List[Tuple[str, str, Dict[str, Any]]]:
    qv = _embed([query])[0]
    res = index.query(vector=qv, top_k=max(1, k), include_metadata=True)
    out = []
    for m in res.matches:
        meta = dict(m.metadata or {})
        text = meta.pop("text", "")
        out.append((m.id, text, meta))
    return out

def export_all():
    # Pinecone has no full-scan API here; stub for parity.
    return []

def reset_all():
    # Fast reset by recreating the index.
    try:
        pc.delete_index(INDEX_NAME)
    except Exception:
        pass
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )
