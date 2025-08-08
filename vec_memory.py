# vec_memory.py
import os, uuid
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# persistent local DB in ./data
os.makedirs("data", exist_ok=True)
client = chromadb.PersistentClient(path="data", settings=Settings(allow_reset=True))

# embeddings via OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env")
embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

COLL = client.get_or_create_collection(
    name="cca_memories",
    embedding_function=embed
)

def upsert_note(text: str, meta: dict | None = None) -> str:
    _id = str(uuid.uuid4())
    COLL.add(documents=[text], metadatas=[meta or {}], ids=[_id])
    return _id

def search(query: str, k: int = 5):
    res = COLL.query(query_texts=[query], n_results=k)
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids   = res.get("ids", [[]])[0]
    return list(zip(ids, docs, metas))

def export_all():
    res = COLL.get(include=["documents","metadatas","ids"])
    return [{"id":i,"text":d,"meta":m} for i,d,m in zip(res["ids"],res["documents"],res["metadatas"])]

def reset_all():
    COLL.delete(where={})  # removes all docs
