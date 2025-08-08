# app.py
import pdf_ingest  # top of file
import json, io
import streamlit as st
from vec_memory import upsert_note, search, export_all, reset_all
from rag_chain import answer

st.set_page_config(page_title="Josh CCA v0", page_icon="🧠", layout="wide")
st.title("🧠 Cognitive Companion Agent v0")

with st.sidebar:
    st.header("Memory Ops")
    note = st.text_area("Add memory / fact")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ingest"):
            if note.strip():
                upsert_note(note.strip(), {"type":"note","source":"ui"})
                st.success("Stored.")
    with colB:
        if st.button("Reset ALL", type="secondary"):
            reset_all()
            st.warning("All vector memories cleared.")

    st.divider()
    # export
    if st.button("Export JSONL"):
        data = export_all()
        buff = io.StringIO()
        for row in data:
            buff.write(json.dumps(row, ensure_ascii=False) + "\n")
        st.download_button("Download memories.jsonl", buff.getvalue(), file_name="memories.jsonl")

    st.subheader("Ingest PDF")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    chunk_size = st.number_input("Chunk size (chars)", min_value=400, max_value=3000, value=1200, step=100)

    if uploaded is not None:
        if st.button("Process PDF"):
            try:
                # show progress (coarse — 3 steps: read → chunk → done)
                prog = st.progress(0, text="Reading PDF…")
                data = uploaded.read()
                prog.progress(33, text="Chunking + storing…")

                total = pdf_ingest.ingest_pdf_bytes(data, uploaded.name, chunk_chars=int(chunk_size))

                prog.progress(100, text="Done ✅")
                st.success(f"Ingested {total} chunks from {uploaded.name}")
            except Exception as e:
                st.error(f"PDF ingest failed: {e}")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat")
    writeback = st.checkbox("Enable write-back of new facts", value=True)
    q = st.text_input("Ask a question")
    if st.button("Ask"):
        if q.strip():
            resp, used = answer(q.strip(), k=5)
            # optionally suppress writeback in answer() (simple: ignore FACT lines here)
            if not writeback:
                st.caption("Write-back disabled for this turn.")
            st.markdown("**Answer:**")
            st.write(resp)
            st.caption(f"Used memory IDs: {used}")

with col2:
    st.subheader("Retrieved Context")
    qq = st.text_input("Preview retrieval for:", value="")
    if st.button("Retrieve"):
        hits = search(qq or "test", k=5)
        for i,(id_,doc,meta) in enumerate(hits,1):
            st.markdown(f"**{i}. id:** `{id_}`")
            st.write(doc[:300] + ("…" if len(doc)>300 else ""))
            if meta: st.caption(meta)
