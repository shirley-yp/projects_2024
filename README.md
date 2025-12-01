# Tiny "Ask-My-Book" RAG Project

This project turns a single book (TXT / Markdown / PDF) into a small **question-answering assistant** that:

- searches the book **by meaning**, not just exact keywords  
- answers using a local LLM (via [Ollama](https://ollama.com/))  
- shows **citations** back to the exact pages / lines it used

It’s a minimal, end-to-end example of **Retrieval-Augmented Generation (RAG)**.

---

## Why this project exists

LLMs are great at language, but they:

- don’t know our **personal books / PDFs**, and  
- often answer without telling us **where** they got the information.

This project is a tiny solution to that:

> “Given *this specific book*, answer my question and show me the exact passages you used.”

Conceptually it does:

1. **Chunk** the document into overlapping pieces (so each chunk holds a complete idea).  
2. **Embed** each chunk into a vector space (so we can search by meaning).  
3. **Store** vectors and metadata in a local vector database (Chroma).  
4. **Retrieve** the top-k most relevant chunks for a question.  
5. **Generate** an answer with a local LLM, using only those chunks and adding citations.

All code tries to stay small and readable so you can use it as a template for other books or internal docs.

---

## Tech stack

- Python 3.9+
- [ChromaDB](https://www.trychroma.com/) for local vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings  
  (`all-MiniLM-L6-v2` by default)
- [Ollama](https://ollama.com/) to run a local LLM  
  (e.g. `llama3.1`, or another model you prefer)
- `pypdf` for PDF text extraction

---

## Project structure

```text
llm_qa_art_of_war/
  src/
    chunk.py            # Step 1–2: read file (txt/md/pdf) → chunks.jsonl
    ingest_chroma.py    # Step 3: embeddings → Chroma collection
    search_chroma.py    # Step 4: debug retrieval, print top-k chunks
    answer_ollama.py    # Step 5: call LLM, answer with citations
  run_pipeline.sh       # Optional: one command to run steps 2–5
  data/
    inputs/             # put your book files here (not tracked by git)
    outputs/            # generated chunks etc. (ignored in git)
    chroma_db/          # Chroma persistent DB (ignored in git)
  README.md
