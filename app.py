import json, io
import streamlit as st
from vec_memory import upsert_note, search, export_all, reset_all
from rag_chain import answer
from ingestors import ingest_pdf_bytes, ingest_txt_bytes, ingest_docx_bytes
from tools import calculator

st.set_page_config(page_title="Josh CCA v0.2", page_icon="🧠", layout="wide")
st.title("🧠 Cognitive Companion Agent v0.2")

with st.sidebar:
    st.header("Memory Ops")
    note = st.text_area("Add memory / fact")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Ingest note") and note.strip():
            upsert_note(note.strip(), {"type":"note","source":"ui"})
            st.success("Stored.")
    with c2:
        if st.button("Reset ALL", type="secondary"):
            reset_all(); st.warning("All vector memories cleared.")

    st.subheader("Ingest Files")
    chunk_size = st.number_input("Chunk size (chars)", min_value=400, max_value=3000, value=1200, step=100)
    upload = st.file_uploader("Upload PDF / TXT / DOCX", type=["pdf","txt","docx"])
    if upload is not None and st.button("Process file"):
        try:
            if upload.type == "application/pdf":
                total = ingest_pdf_bytes(upload.read(), upload.name, int(chunk_size))
            elif upload.type in ("text/plain",):
                total = ingest_txt_bytes(upload.read(), upload.name, int(chunk_size))
            else:
                total = ingest_docx_bytes(upload.read(), upload.name, int(chunk_size))
            st.success(f"Ingested {total} chunks from {upload.name}")
        except Exception as e:
            st.error(f"Ingest failed: {e}")

    st.subheader("Export")
    if st.button("Export JSONL"):
        data = export_all()
        buff = io.StringIO()
        for row in data: buff.write(json.dumps(row, ensure_ascii=False) + "\n")
        st.download_button("Download memories.jsonl", buff.getvalue(), file_name="memories.jsonl")

    st.subheader("Quick Calculator")
    expr = st.text_input("Expression (e.g., 12*(5+3)/4)")
    if st.button("Compute") and expr.strip():
        st.write(calculator(expr))

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat")
    q = st.text_input("Ask a question")
    if st.button("Ask") and q.strip():
        resp, used = answer(q.strip(), k=5)
        st.markdown("**Answer:**"); st.write(resp)
        st.caption(f"Used memory IDs: {used}")

with col2:
    st.subheader("Retrieved Context Preview")
    qq = st.text_input("Preview retrieval for:", value="")
    if st.button("Retrieve"):
        hits = search(qq or "test", k=5)
        for i,(id_,doc,meta) in enumerate(hits,1):
            st.markdown(f"**{i}. id:** `{id_}`")
            st.write((doc[:300] + "…") if len(doc)>300 else doc)
            if meta: st.caption(meta)