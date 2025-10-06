import re, os, tempfile, pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss
from sklearn.preprocessing import normalize

# ---------- PDF ----------
def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- Text chunking ----------
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c.strip()]

def make_safe_chunks(pages, tokenizer, chunk_size_chars=800, overlap_chars=200, max_tokens=500):
    raw = []
    for p in pages:
        raw.extend(chunk_text(p, chunk_size_chars, overlap_chars))
    safe = []
    for r in raw:
        tokens = tokenizer.tokenize(r)
        if len(tokens) <= max_tokens:
            safe.append(r)
        else:
            for i in range(0, len(tokens), max_tokens):
                sub = tokens[i:i+max_tokens]
                safe.append(tokenizer.convert_tokens_to_string(sub))
    return safe

# ---------- FAISS ----------
class FaissStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vectors, metadatas):
        self.index.add(vectors.astype('float32'))
        self.metadatas.extend(metadatas)

    def search(self, q_vec, top_k=4):
        q = q_vec.astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        D, I = self.index.search(q, top_k)
        results = []
        for j, idx in enumerate(I[0]):
            if idx < len(self.metadatas):
                results.append((self.metadatas[idx], float(D[0][j])))
        return results

# ---------- Prompt ----------
def build_prompt(question, retrieved_texts, max_context_chars=3000):
    context = "\n\n".join(retrieved_texts)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    return f"Answer the question using ONLY the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# ---------- RAG pipeline ----------
def build_rag_system(embed_model_name, gen_model_name, pdf_path, chunk_size=800, overlap=200):
    embed_model = SentenceTransformer(embed_model_name)
    gen_model = pipeline("text2text-generation", model=gen_model_name, device=-1)

    pages = pdf_to_text(pdf_path)
    pages = [clean_text(p) for p in pages if p.strip()]

    list_of_chunks = make_safe_chunks(pages, embed_model.tokenizer, chunk_size, overlap)
    vectors = embed_model.encode(list_of_chunks, convert_to_numpy=True, show_progress_bar=True)
    vectors = normalize(vectors, axis=1)

    fa = FaissStore(vectors.shape[1])
    metas = [{"text": c, "source": os.path.basename(pdf_path)} for c in list_of_chunks]
    fa.add(vectors, metas)

    return embed_model, gen_model, fa, list_of_chunks

def ask_question(question, embed_model, gen_model, fa_store, top_k=3):
    q_vec = embed_model.encode([question], convert_to_numpy=True)
    q_vec = normalize(q_vec, axis=1)
    retrieved = fa_store.search(q_vec, top_k=top_k)
    retrieved_texts = [r[0]["text"] for r in retrieved]
    prompt = build_prompt(question, retrieved_texts)
    out = gen_model(prompt, max_length=256, do_sample=False)
    answer = out[0]["generated_text"]
    return answer, retrieved_texts
