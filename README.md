---
title: GenAI PDF Q&A Chatbot (Groq)
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# GenAI PDF Q&A Chatbot (RAG · LangChain · FAISS · **Groq**)

Upload PDFs → build a vector index → ask grounded questions with page-level citations.  
This Space uses **LangChain**, **FAISS**, and **Groq** LLMs (e.g., `llama-3.1-70b-versatile`).  
If `GROQ_API_KEY` isn’t set, it falls back to OpenAI (if provided) or shows a helpful message.

---

## 🚀 Features
- 📥 Upload multiple PDFs
- 🔪 Smart chunking (`RecursiveCharacterTextSplitter`)
- 🧠 Embeddings:  
  - **OpenAI** `text-embedding-3-small` when `OPENAI_API_KEY` is set  
  - Fallback: **sentence-transformers/all-MiniLM-L6-v2** (CPU-friendly, no key)
- 🗃️ Vector store: **FAISS** (persisted to `storage/index/`)
- 💬 LLM: **Groq** via `langchain_groq` (default), or OpenAI if configured
- 🔎 Citations: filename + page (e.g., `[source: myfile.pdf p.3]`)

---

## ✅ Quick Start (on this Space)
1. **Add secrets** (Space → *Settings* → *Variables and secrets*):
   - `GROQ_API_KEY` = *your Groq key*  ✅  
   - *(optional)* `GROQ_MODEL` = `llama-3.1-70b-versatile` (default)  
   - *(optional)* `OPENAI_API_KEY` = *your OpenAI key* (enables OpenAI embeddings; otherwise uses local sentence-transformers)
2. Wait for the Space to rebuild.
3. Open the app:
   - Upload 1+ PDFs
   - Click **🔧 Build Index**
   - Ask a question with **💬 Ask**

> **Note:** If your PDFs are scans (images), text may not be extractable. Run OCR first.

---

## 🔧 Tech Stack
- **UI**: Gradio Blocks
- **RAG**: LangChain + FAISS
- **Embeddings**: OpenAI or Sentence-Transformers (fallback)
- **LLM**: Groq (`langchain_groq`) — default model `llama-3.1-70b-versatile`

---

## 🗂️ Project Structure
```
app.py                # Gradio UI + RAG pipeline
requirements.txt      # Python deps for Spaces
README.md             # This file (with Spaces front-matter)
storage/              # created at runtime to store FAISS index
```

---

## 🔐 Secrets (recap)
- `GROQ_API_KEY` **(required for Groq)**
- `GROQ_MODEL` *(optional)* → e.g., `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`
- `OPENAI_API_KEY` *(optional)* → used for OpenAI embeddings or fallback LLM

---

## 🧩 How It Works
1. **Load PDFs** with `PyPDFLoader` (keeps `source` & `page` in metadata)
2. **Split** into ~900-char chunks with 150 overlap
3. **Embed** chunks (OpenAI if key; else `all-MiniLM-L6-v2`)
4. **Index** with FAISS → saved to `storage/index/`
5. **Retrieve** top-k chunks for the query (k=4)
6. **Generate** answer with Groq LLM (grounded prompt: “If not in docs, say you don’t know”)
7. **Cite** sources as `[source: filename.pdf p.#]`

---

## 🧪 Local Dev (optional)
```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:7860
```

---

## 🩹 Troubleshooting
- **Configuration error (this page)** → Make sure the YAML front-matter at the top of this README matches:
  - `sdk: gradio`, `app_file: app.py`, and a valid `sdk_version` (e.g., `"4.44.0"`)
- **Build fails** → Check `requirements.txt` includes:
  ```
  gradio
  langchain
  langchain-community
  langchain-openai
  langchain-groq
  groq
  faiss-cpu
  sentence-transformers
  pypdf
  tiktoken
  numpy
  ```
- **“No LLM provider configured”** → Add `GROQ_API_KEY` in *Variables & secrets*
- **Empty/irrelevant answers** → Ensure you clicked **🔧 Build Index** after uploading PDFs
- **Scanned PDFs** → Use OCR before uploading

---

## 📄 License
Choose the license that fits your needs (e.g., MIT).
