#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import textwrap

import chromadb
from chromadb.config import Settingss


def format_sources(results):
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    blocks = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        src = meta.get("source_file", "unknown")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")
        sp = meta.get("start_page")
        ep = meta.get("end_page")

        if sp is not None and ep is not None:
            header = f"[{rank}] {src} pages={sp}-{ep} lines={sl}-{el} (distance={dist:.4f}, id={cid})"
        else:
            header = f"[{rank}] {src} lines={sl}-{el} (distance={dist:.4f}, id={cid})"
        blocks.append(header + "\n" + doc.strip())
    return "\n\n---\n\n".join(blocks)


def call_ollama(model: str, prompt: str) -> str:
    # Requires: `ollama` installed and running
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout.decode("utf-8", errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", required=True)
    ap.add_argument("--collection", default="art_of_war")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("question", nargs="+")
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    db_dir = Path(args.db_dir)

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=args.collection)

    res = collection.query(
        query_texts=[question],
        n_results=args.k,
        include=["documents", "metadatas", "distances"],
    )

    sources_text = format_sources(res)

    prompt = textwrap.dedent(f"""
    You are answering questions using ONLY the provided sources from a book.
    Rules:
    - If the sources do not contain the answer, say: "Not found in the provided sources."
    - Do not invent facts.
    - Write a concise answer (5-8 sentences).
    - Add citations after claims using the format:
      - TXT/MD: (source_file:lines start-end)
      - PDF: (source_file:pages start-end, lines start-end)
    - Do NOT paste long quotes; keep quotes under 20 words.

    Question:
    {question}

    Sources:
    {sources_text}

    Now write the answer with citations.
    """).strip()

    answer = call_ollama(args.model, prompt)

    # Print
    print("\n==============================")
    print("QUESTION:", question)
    print("==============================\n")
    print(answer.strip())
    print("\n------------------------------")
    print("SOURCES USED (top-k retrieved):")
    for meta in res["metadatas"][0]:
        src = meta.get("source_file", "unknown")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")
        sp = meta.get("start_page")
        ep = meta.get("end_page")

        if sp is not None and ep is not None:
            print(f"- {src} pages={sp}-{ep} lines={sl}-{el}")
        else:
            print(f"- {src} lines={sl}-{el}")
    print("------------------------------\n")


if __name__ == "__main__":
    main()
