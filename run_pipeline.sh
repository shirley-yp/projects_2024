#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./run_pipeline.sh "<input_path>" "<collection_name>" "<question>"
#
# Example:
# ./run_pipeline.sh data/inputs/art_of_war.txt art_of_war "What does it say about deception?"

INPUT="${1:?input file path required}"
COLLECTION="${2:?collection name required}"
QUESTION="${3:?question required}"

# Resolve project root = folder where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"

CODE="$ROOT/src"
OUT_DIR="$ROOT/data/outputs"
DB_DIR="$ROOT/data/chroma_db"

CHUNKS_JSONL="$OUT_DIR/${COLLECTION}_chunks.jsonl"

# If you use a venv, activate it manually before calling this script:
#   source .venv/bin/activate
# So we don't activate anything here.

export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR" "$DB_DIR"

echo "== Step 2: chunking =="
python "$CODE/chunk.py" "$INPUT" \
  --out "$CHUNKS_JSONL" \
  --lines_per_chunk 80 \
  --overlap_lines 20

echo "== Step 3: ingest =="
python "$CODE/ingest_chroma.py" \
  --chunks "$CHUNKS_JSONL" \
  --db_dir "$DB_DIR" \
  --collection "$COLLECTION"

echo "== Step 5: answer (with sources) =="
python "$CODE/answer_ollama.py" \
  --db_dir "$DB_DIR" \
  --collection "$COLLECTION" \
  --k 5 \
  "$QUESTION"