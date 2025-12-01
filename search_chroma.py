#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", required=True, help="Chroma DB directory")
    ap.add_argument("--collection", default="art_of_war")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")  # must match ingest model
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("question", nargs="+", help="Question text")
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

    print("\n==============================")
    print("QUESTION:", question)
    print("==============================\n")

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]  # lower = closer for Chroma distance metrics

    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        src = meta.get("source_file", "unknown")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")
        sp = meta.get("start_page")  # may be None for txt/md
        ep = meta.get("end_page")

        print(f"[{rank}] id={cid}  distance={dist:.4f}")

        # Print a single, consistent citation line
        if sp is not None and ep is not None:
            print(f"    source={src} pages={sp}-{ep} lines={sl}-{el}")
        else:
            print(f"    source={src} lines={sl}-{el}")
        preview = doc.strip().replace("\n", " ")
        if len(preview) > 320:
            preview = preview[:320] + "..."
        print(f"    text: {preview}\n")


if __name__ == "__main__":
    main()
