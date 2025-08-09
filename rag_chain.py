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

SYS = ("You are Josh's Cognitive Companion. Use CONTEXT if relevant. "
       "If arithmetic is needed, use the calculator tool. Be concise; if info is missing, ask ONE clarifying question. "
       "After answering, list any new facts as lines starting with 'FACT:'.")

def _decide_calc(q: str) -> bool:
    m = [SystemMessage(content="Reply YES or NO only."),
         HumanMessage(content=f"Does this question require arithmetic? {q}")]
    return llm.invoke(m).content.strip().upper().startswith("Y")

def _extract_expr(q: str) -> str:
    m = [SystemMessage(content="Return only a bare arithmetic expression."),
         HumanMessage(content=f"Extract minimal arithmetic expression from: {q}")]
    return llm.invoke(m).content.strip()

def answer(query: str, k: int = 5):
    hits = search(query, k)
    ctx = "\n".join(f"[{i}] {doc}" for i, (_, doc, _) in enumerate(hits))

    tool_note = ""
    if _decide_calc(query):
        expr = _extract_expr(query)
        tool_note = f"\n\nTOOL_RESULT:\n{calculator(expr)}"

    msgs = [
        SystemMessage(content=SYS),
        HumanMessage(content=f"CONTEXT:\n{ctx}\n{tool_note}\n\nQUESTION: {query}\nReply, then list new facts as 'FACT: ...'")
    ]
    resp = llm.invoke(msgs).content

    for line in resp.splitlines():
        if line.strip().upper().startswith("FACT:"):
            upsert_note(line[5:].strip(), {"type":"fact","source":"writeback"})
    used_ids = [h[0] for h in hits]
    return resp, used_ids
