
import os
import tempfile
import shutil
from typing import List, Tuple

import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Groq (LLM) via LangChain
try:
    from langchain_groq import ChatGroq
    _HAS_GROQ = True
except Exception:
    _HAS_GROQ = False

# Optional: local sentence-transformers if no OpenAI embeddings available
from sentence_transformers import SentenceTransformer

SPACE_STORAGE_DIR = "storage"
os.makedirs(SPACE_STORAGE_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a helpful assistant for question answering over provided PDF documents.
Use only the retrieved context to answer. If the answer isn't in the documents, say you don't know.
When helpful, cite sources as [source: filename.pdf p.#] using the provided metadata.
Keep answers concise and accurate."""

# ---- Embeddings Helper ----
class SafeEmbedding(Embeddings):
    """
    Prefers OpenAI Embeddings if OPENAI_API_KEY is available; otherwise uses a local model.
    """
    def __init__(self):
        self.use_openai = bool(os.getenv("OPENAI_API_KEY"))
        if self.use_openai:
            self._openai = OpenAIEmbeddings(model="text-embedding-3-small")
            self.dim = 1536
        else:
            # small, CPU-friendly model
            self._local = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.dim = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self, "_openai"):
            return self._openai.embed_documents(texts)
        return self._local.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        if hasattr(self, "_openai"):
            return self._openai.embed_query(text)
        vec = self._local.encode([text], convert_to_numpy=True)[0]
        return vec.tolist()

def load_and_split_pdfs(file_paths: List[str]) -> List[Document]:
    docs = []
    for fp in file_paths:
        loader = PyPDFLoader(fp)
        file_docs = loader.load()
        for d in file_docs:
            base = os.path.basename(fp)
            d.metadata["source"] = base
        docs.extend(file_docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_documents(docs)

def build_vectorstore(docs: List[Document], embedding: Embeddings) -> FAISS:
    return FAISS.from_documents(docs, embedding)

def format_citations(docs: List[Document]) -> str:
    cites = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page", None)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            cites.append(f"[source: {src} p.{page+1}]")
        else:
            cites.append(f"[source: {src}]")
    return " ".join(cites)

def make_rag_chain(vs: FAISS):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    # Prefer Groq if key present and package available
    if os.getenv("GROQ_API_KEY") and _HAS_GROQ:
        # Good default Groq models: llama-3.1-70b-versatile / mixtral-8x7b-32768 / gemma2-9b-it
        model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        llm = ChatGroq(model_name=model, temperature=0)
        parser = StrOutputParser()
        def chain(question: str, context: str) -> str:
            msg = prompt.format_messages(question=question, context=context)
            return parser.parse(llm.invoke(msg))
        return retriever, chain

    # Fallback to OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        parser = StrOutputParser()
        def chain(question: str, context: str) -> str:
            msg = prompt.format_messages(question=question, context=context)
            return parser.parse(llm.invoke(msg))
        return retriever, chain

    # If no provider set
    def offline_chain(question, context):
        return ("No LLM provider configured. Add a GROQ_API_KEY (preferred) or OPENAI_API_KEY "
                "as a Space secret to generate answers.")
    return retriever, offline_chain

def ingest(files: List[gr.File]) -> Tuple[str, str]:
    if not files:
        return "Please upload at least one PDF.", ""

    tmpdir = tempfile.mkdtemp()
    saved_paths = []
    for f in files:
        if not f.name.lower().endswith(".pdf"):
            continue
        dest = os.path.join(tmpdir, os.path.basename(f.name))
        shutil.copyfile(f.name, dest)
        saved_paths.append(dest)

    if not saved_paths:
        return "No PDFs found in upload.", ""

    docs = load_and_split_pdfs(saved_paths)
    embedding = SafeEmbedding()
    vs = build_vectorstore(docs, embedding)

    faiss_path = os.path.join(SPACE_STORAGE_DIR, "index")
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    vs.save_local(faiss_path)

    return f"Ingested {len(saved_paths)} PDFs · {len(docs)} chunks · Embeddings dim={embedding.dim}", faiss_path

def answer_question(question: str, faiss_path: str) -> Tuple[str, str]:
    if not question or not question.strip():
        return "Ask a question about your PDFs.", ""
    if not faiss_path or not os.path.exists(faiss_path):
        return "Please upload PDFs and click 'Build Index' first.", ""

    embedding = SafeEmbedding()
    vs = FAISS.load_local(faiss_path, embedding, allow_dangerous_deserialization=True)

    retriever, rag_call = make_rag_chain(vs)

    retrieved = retriever.get_relevant_documents(question)
    if not retrieved:
        return "No relevant context found in the PDFs.", ""

    context = "\n\n---\n\n".join([d.page_content for d in retrieved][:6])
    cites = format_citations(retrieved)

    output = rag_call(question, context)

    final = output.strip()
    if cites:
        final += "\n\n" + cites
    return final, cites

with gr.Blocks(title="GenAI PDF Q&A (RAG · LangChain)") as demo:
    gr.Markdown(
        """
        # 📄 GenAI PDF Q&A — RAG on LangChain
        Upload PDFs → Build an index → Ask questions.  
        **Embeddings:** OpenAI (fallback to sentence-transformers) · **Vector DB:** FAISS · **LLM:** Groq (preferred) or OpenAI

        > Tip: On Hugging Face Spaces, add a `GROQ_API_KEY` (or `OPENAI_API_KEY`) [Secret].
        """
    )

    with gr.Row():
        file_uploader = gr.File(label="Upload one or more PDF files", file_types=[".pdf"], file_count="multiple")
    with gr.Row():
        build_btn = gr.Button("🔧 Build Index", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        faiss_path_state = gr.State("")

    with gr.Row():
        question = gr.Textbox(label="Ask a question", placeholder="e.g., Summarize chapter 2's main points")
    with gr.Row():
        ask_btn = gr.Button("💬 Ask", variant="primary")
    with gr.Row():
        answer = gr.Markdown()
    with gr.Row():
        sources = gr.Textbox(label="Citations", interactive=False)

    def _build(files):
        msg, path = ingest(files)
        return msg, path

    def _ask(q, path):
        ans, cites = answer_question(q, path)
        return ans, cites

    build_btn.click(_build, inputs=[file_uploader], outputs=[status, faiss_path_state])
    ask_btn.click(_ask, inputs=[question, faiss_path_state], outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
