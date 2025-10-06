import streamlit as st
from rag_backend import build_rag_system, ask_question
import tempfile

st.set_page_config(page_title="📚 RAG PDF Q&A", layout="wide")
st.title("📚 RAG PDF Question Answering System")

with st.sidebar:
    st.header("⚙️ Settings")
    embed_choice = st.selectbox("Embedding model", [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ])
    gen_choice = st.selectbox("Generation model", [
        "google/flan-t5-base",
        "google/flan-t5-large"
    ])
    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 800, 100)
    overlap = st.slider("Overlap size (characters)", 50, 500, 200, 50)
    top_k = st.slider("Top-k retrieved chunks", 1, 8, 3)

uploaded = st.file_uploader("📂 Upload your PDF file", type=["pdf"])

if uploaded:
    st.success("✅ PDF uploaded successfully!")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        tmpf.write(uploaded.getbuffer())
        pdf_path = tmpf.name

    with st.spinner("🔄 Building RAG system... This may take a minute..."):
        embed_model, gen_model, fa_store, chunks = build_rag_system(
            embed_choice, gen_choice, pdf_path, chunk_size, overlap
        )
    st.success(f"✅ Processed {len(chunks)} chunks!")

    question = st.text_input("❓ Ask a question based on the PDF content:")
    if question:
        with st.spinner("🧠 Thinking... generating answer..."):
            answer, retrieved_texts = ask_question(question, embed_model, gen_model, fa_store, top_k)
        st.subheader("💬 Answer:")
        st.write(answer)

        st.subheader("📄 Retrieved context:")
        for i, txt in enumerate(retrieved_texts, 1):
            st.markdown(f"**Chunk {i}:** {txt[:800]}...")
else:
    st.info("📥 Please upload a PDF file to start.")
