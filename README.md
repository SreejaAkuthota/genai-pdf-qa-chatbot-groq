
# GenAI PDF Q&A (RAG) — Groq + LangChain + FAISS on Hugging Face Spaces

**Live-ready** PDF Q&A chatbot. Uses **Groq** LLMs (e.g., `llama-3.1-70b-versatile`) via `langchain_groq`. Falls back to **OpenAI** if provided, and to local **sentence-transformers** for embeddings when no OpenAI key.

## Deploy on Hugging Face Spaces
1. Create Space (Gradio, CPU).
2. Upload `app.py`, `requirements.txt`, `README.md`.
3. Add Secrets:
   - `GROQ_API_KEY`: your Groq key (preferred)
   - Optional: `GROQ_MODEL` (default `llama-3.1-70b-versatile`)
   - Optional: `OPENAI_API_KEY` (for embeddings or fallback LLM)
4. Run: upload PDFs → 🔧 Build Index → 💬 Ask.

## Notes
- Citations include filename + page.
- FAISS is persisted under `storage/index/`.
- You can swap FAISS with Chroma/Pinecone; edit `build_vectorstore` and loader.
