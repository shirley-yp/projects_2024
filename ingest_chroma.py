#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--db_dir", required=True, help="Directory to store Chroma DB")
    ap.add_argument("--collection", default="art_of_war")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from: {chunks_path}")
    records = load_jsonl(chunks_path)
    print(f"Loaded {len(records)} chunks")

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Opening Chroma DB at: {db_dir}")
    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=args.collection)

    # Prepare data
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metadatas = []
    for r in records:
        m = {
            "source_file": r["source_file"],
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
        }
        # NEW: include PDF page range if available
        if "start_page" in r and "end_page" in r:
            m["start_page"] = int(r["start_page"])
            m["end_page"] = int(r["end_page"])
        metadatas.append(m)


    # Embed + insert in batches
    total = len(texts)
    batch = args.batch_size
    for i in range(0, total, batch):
        j = min(i + batch, total)
        batch_ids = ids[i:j]
        batch_texts = texts[i:j]
        batch_metas = metadatas[i:j]

        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metas,
        )
        print(f"Inserted {j}/{total}")

    print("\nDone ✅")
    print(f"Collection: {args.collection}")
    print(f"DB dir: {db_dir}")
    print(f"Count in collection: {collection.count()}")


if __name__ == "__main__":
    main()
