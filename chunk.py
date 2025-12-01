#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def read_text_any(src: Path) -> Tuple[str, Optional[List[int]]]:
    """
    Returns:
      text: extracted text
      page_of_line: for PDFs, a list mapping each line index -> page number (1-based).
                    for non-PDFs, returns None.
    """
    suffix = src.suffix.lower()

    if suffix in [".txt", ".md"]:
        text = src.read_text(encoding="utf-8", errors="ignore")
        return text, None

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(src))

        parts: List[str] = []
        page_of_line: List[int] = []

        # We generate lines + map each line to a PDF page number.
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
            lines = page_text.split("\n")

            # Add a visible page marker line (helps debugging/citations)
            parts.append(f"[[PAGE {i}]]\n")
            page_of_line.append(i)

            for ln in lines:
                parts.append(ln + "\n")
                page_of_line.append(i)

            parts.append("\n")  # page separator
            page_of_line.append(i)

        return "".join(parts), page_of_line

    raise ValueError(f"Unsupported file type: {src.suffix} (use .txt, .md, or .pdf)")

def normalize_lines(text: str) -> List[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                normalized.append("")
        else:
            blank_run = 0
            normalized.append(ln.rstrip())
    return normalized

def chunk_by_lines(
    lines: List[str],
    lines_per_chunk: int = 80,
    overlap_lines: int = 20,
) -> List[Tuple[int, int, str]]:
    assert lines_per_chunk > 0
    assert 0 <= overlap_lines < lines_per_chunk

    chunks: List[Tuple[int, int, str]] = []
    i = 0
    n = len(lines)
    step = lines_per_chunk - overlap_lines

    while i < n:
        start = i
        end = min(i + lines_per_chunk, n)  # end is exclusive
        chunk_text = "\n".join(lines[start:end]).strip()
        if chunk_text:
            chunks.append((start + 1, end, chunk_text))  # 1-indexed
        i += step

    return chunks

def build_records(
    source_path: Path,
    chunks: List[Tuple[int, int, str]],
    page_of_line: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, (start_line, end_line, text) in enumerate(chunks):
        rec: Dict[str, Any] = {
            "id": f"{source_path.stem}__chunk_{idx:05d}",
            "source_file": source_path.name,
            "start_line": start_line,
            "end_line": end_line,
            "text": text,
        }
        # Add page range for PDFs if we have a mapping
        if page_of_line is not None and len(page_of_line) >= end_line:
            rec["start_page"] = int(page_of_line[start_line - 1])
            rec["end_page"] = int(page_of_line[end_line - 1])
        records.append(rec)
    return records

def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to .txt/.md/.pdf file")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--lines_per_chunk", type=int, default=80)
    ap.add_argument("--overlap_lines", type=int, default=20)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out)

    text, page_of_line = read_text_any(src)
    lines = normalize_lines(text)
    chunks = chunk_by_lines(lines, args.lines_per_chunk, args.overlap_lines)
    records = build_records(src, chunks, page_of_line=page_of_line)
    write_jsonl(records, out)

    print(f"Source: {src} ({len(lines)} lines)")
    print(f"Chunks: {len(records)} → {out}")
    if records:
        ex = records[0]
        keys = ["id", "source_file", "start_line", "end_line"]
        if "start_page" in ex:
            keys += ["start_page", "end_page"]
        print("Example record:")
        print({k: ex[k] for k in keys})
        print(ex["text"][:300] + ("..." if len(ex["text"]) > 300 else ""))

if __name__ == "__main__":
    main()
