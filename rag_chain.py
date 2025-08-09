import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from vec_memory import search, upsert_note
from tools import calculator

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

SYS = (
    "You are Josh's Cognitive Companion. Use CONTEXT if relevant. "
    "If TOOL_RESULT is present, incorporate it. Be concise; if info is missing, ask ONE clarifying question. "
    "After answering, list any new facts as lines starting with 'FACT:'."
)

def _decide_calc(q: str) -> bool:
    # Heuristic for when to try calculator
    triggers = ["+", "-", "*", "/", "%", "^", "sum", "total", "difference", "times", "multiply", "divide", "add", "subtract"]
    return any(t in q for t in triggers)

def answer(query: str, k: int = 5):
    hits = search(query, k)
    ctx = "\n".join(f"[{i}] {doc}" for i, (_, doc, _) in enumerate(hits))

    tool_note = calculator(query) if _decide_calc(query) else ""

    msgs = [
        SystemMessage(content=SYS),
        HumanMessage(content=f"CONTEXT:\n{ctx}\n\nTOOL_RESULT:\n{tool_note}\n\nQUESTION: {query}\nReply, then list new facts as 'FACT: ...'")
    ]
    resp = llm.invoke(msgs).content

    for line in resp.splitlines():
        if line.strip().upper().startswith("FACT:"):
            upsert_note(line[5:].strip(), {"type": "fact", "source": "writeback"})

    used_ids = [h[0] for h in hits]
    return resp, used_ids
