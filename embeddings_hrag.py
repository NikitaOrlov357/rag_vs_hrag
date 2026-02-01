import json
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


# ======================================================
# PATHS
# ======================================================

DATA_DIR = Path(r"C:\Users\mywke\PythonProjects\isp\data")
OUTPUT_DIR = Path(r"C:\Users\mywke\PythonProjects\isp\embeddings_hrag")

PARENTS_OUT = OUTPUT_DIR / "parents.jsonl"
CHILDREN_OUT = OUTPUT_DIR / "children.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 200

# parent aggregation text cap (avoid huge monthly blobs)
MAX_PARENT_TEXT_CHARS = 25000

# ======================================================
# UTILS
# ======================================================

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_text(item: dict) -> str:
    title = item.get("title", "")
    summary = item.get("summary", "")
    return f"{title}\n\n{summary}".strip()

# data\2024\april_2024.jsonl -> year=2024, month=april
def parse_year_month(src_path: Path):
    year = src_path.parent.name
    stem = src_path.stem.lower()
    m = re.match(r"([a-z]+)_\d{4}", stem)
    month = m.group(1) if m else stem
    return year, month

def chunk_text(text: str, size: int, overlap: int):
    text = text.strip()
    if not text:
        return []
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks

def build_parent_text(year: str, month: str, items: list) -> str:
    header = f"YEAR: {year}\nMONTH: {month}\n"
    parts = []
    total = 0
    for it in items:
        t = build_text(it)
        if not t:
            continue
        if total + len(t) + 2 > MAX_PARENT_TEXT_CHARS:
            break
        parts.append(t)
        total += len(t) + 2
    return (header + "\n\n".join(parts)).strip()





def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    jsonl_files = sorted(DATA_DIR.rglob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl files found.")
        return

    print(f"Found {len(jsonl_files)} JSONL files")

    parents_rows = []
    children_rows = []

    parent_texts = []
    parent_meta = []

    child_texts = []
    child_meta = []

    for src_path in jsonl_files:
        year, month = parse_year_month(src_path)
        parent_id = f"{year}-{month}"
        rel_path = src_path.relative_to(DATA_DIR)

        print("\n----------------------------------------")
        print(f"Input : {src_path}")
        print(f"Group : parent_id={parent_id}")
        print(f"Rel   : {rel_path}")

        data = list(read_jsonl(src_path))
        if not data:
            print("  Skipped (empty file)")
            continue

        # Parent (year-month)
        p_text = build_parent_text(year, month, data)
        parent_texts.append(p_text)
        parent_meta.append({
            "parent_id": parent_id,
            "year": year,
            "month": month,
            "source_file": str(src_path),
            "relative_file": str(rel_path),
            "doc_count": len(data),
        })

        # Children
        for idx, item in enumerate(data):
            doc_id = str(item.get("id", f"{parent_id}-{idx:04d}"))

            base_text = build_text(item)
            chunks = chunk_text(base_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
            if not chunks:
                continue

            for cidx, ctext in enumerate(chunks):
                child_id = f"{doc_id}::c{cidx}"
                child_texts.append(ctext)
                child_meta.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "chunk_index": cidx,
                    "date": item.get("date", ""),
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source_file": str(src_path),
                    "relative_file": str(rel_path),
                    "chunk_text": ctext,
                })

        print(f"  Loaded docs: {len(data)} | child chunks so far: {len(child_texts)}")

    print("\n========================================")
    print(f"Parents to embed : {len(parent_texts)}")
    print(f"Children to embed: {len(child_texts)}")

    # Parents
    parent_embs = model.encode(
        parent_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for meta, emb in zip(parent_meta, parent_embs):
        row = dict(meta)
        row["embedding_model"] = EMBEDDING_MODEL
        row["embedding_dim"] = int(emb.shape[0])
        row["embedding"] = emb.tolist()
        parents_rows.append(row)

    # Childrens
    child_embs = model.encode(
        child_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for meta, emb in zip(child_meta, child_embs):
        row = dict(meta)
        row["embedding_model"] = EMBEDDING_MODEL
        row["embedding_dim"] = int(emb.shape[0])
        row["embedding"] = emb.tolist()
        children_rows.append(row)

    # outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PARENTS_OUT, parents_rows)
    write_jsonl(CHILDREN_OUT, children_rows)

    print("\n========================================")
    print("All hierarchical embeddings generated successfully.")
    print(f"Parents  -> {PARENTS_OUT} ({len(parents_rows)})")
    print(f"Children -> {CHILDREN_OUT} ({len(children_rows)})")
    print("Embeddings are normalized → use cosine similarity via inner product.")




if __name__ == "__main__":
    main()
