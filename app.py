# app.py
import streamlit as st
from vec_memory import upsert_note, search
from rag_chain import answer

st.set_page_config(page_title="Josh CCA v0", page_icon="🧠", layout="wide")
st.title("🧠 Cognitive Companion Agent v0")

with st.sidebar:
    st.header("Add Memory")
    note = st.text_area("Write a memory/fact/note to store")
    if st.button("Ingest"):
        if note.strip():
            upsert_note(note.strip(), {"type":"note","source":"ui"})
            st.success("Stored.")
        else:
            st.warning("Enter some text first.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat")
    q = st.text_input("Ask a question")
    if st.button("Ask"):
        if q.strip():
            resp, used = answer(q.strip(), k=5)
            st.markdown("**Answer:**")
            st.write(resp)
            st.caption(f"Used memory IDs: {used}")
        else:
            st.warning("Type a question.")

with col2:
    st.subheader("Quick Search")
    qq = st.text_input("Search memories")
    if st.button("Search"):
        if qq.strip():
            hits = search(qq.strip(), k=5)
            for i,(id_,doc,meta) in enumerate(hits,1):
                st.write(f"{i}. {doc}")
        else:
            st.warning("Enter a search query.")
