# rag_chain.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from vec_memory import search, upsert_note

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

SYS = ("You are Josh's Cognitive Companion. Use CONTEXT if relevant. "
       "Be concise. If info is missing, ask one clarifying question. "
       "After answering, propose 0–3 new atomic facts as lines starting with 'FACT:'")

def answer(query: str, k: int = 5):
    hits = search(query, k)
    ctx = "\n".join(f"[{i}] {doc}" for i, (_, doc, _) in enumerate(hits))
    msgs = [
        SystemMessage(content=SYS),
        HumanMessage(content=f"CONTEXT:\n{ctx}\n\nQUESTION: {query}\n—\nReply, then list any new facts as 'FACT: ...'")
    ]
    resp = llm.invoke(msgs).content

    # write back any proposed facts
    for line in resp.splitlines():
        if line.strip().upper().startswith("FACT:"):
            upsert_note(line[5:].strip(), {"type":"fact","source":"writeback"})
    used_ids = [h[0] for h in hits]
    return resp, used_ids
