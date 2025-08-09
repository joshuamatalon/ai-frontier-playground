import os, uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

DATA_DIR = "data"
COLL_NAME = "cca_memories"
os.makedirs(DATA_DIR, exist_ok=True)

# Always load the .env that sits beside this file
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)  # fallback to process env if missing

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(f"OPENAI_API_KEY missing in .env at {ENV_PATH} or environment")

client = chromadb.PersistentClient(path=DATA_DIR, settings=Settings(allow_reset=True))
_lc_emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

class LCEmbeddingFn:
    def __call__(self, input):             # Chroma expects param named 'input'
        return _lc_emb.embed_documents(list(input))
    def name(self):
        return "langchain_openai_adapter"

embed_fn = LCEmbeddingFn()
COLL = client.get_or_create_collection(name=COLL_NAME, embedding_function=embed_fn)

def upsert_note(text: str, meta: Dict[str, Any] | None = None) -> str:
    _id = str(uuid.uuid4())
    COLL.add(documents=[text], metadatas=[meta or {}], ids=[_id])
    return _id

def upsert_many(chunks: List[str], meta: Dict[str, Any]) -> List[str]:
    if not chunks:
        return []
    ids = [str(uuid.uuid4()) for _ in chunks]
    COLL.add(documents=chunks, metadatas=[meta] * len(chunks), ids=ids)
    return ids

def search(query: str, k: int = 5) -> List[Tuple[str, str, Dict[str, Any]]]:
    res = COLL.query(query_texts=[query], n_results=max(1, k))
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    return list(zip(ids, docs, metas))

def export_all():
    res = COLL.get(include=["documents", "metadatas", "ids"])
    return [
        {"id": i, "text": d, "meta": m}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]

def reset_all():
    COLL.delete(where={})

