import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer


# ======================================================
# PATHS
# ======================================================

DATA_DIR = Path(r"C:\Users\mywke\PythonProjects\isp\data")
OUTPUT_DIR = Path(r"C:\Users\mywke\PythonProjects\isp\embeddings_rag")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

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

    for src_path in jsonl_files:
        rel_path = src_path.relative_to(DATA_DIR)
        out_path = OUTPUT_DIR / rel_path

        print("\n----------------------------------------")
        print(f"Input : {src_path}")
        print(f"Output: {out_path}")

        data = list(read_jsonl(src_path))
        if not data:
            print("  Skipped (empty file)")
            continue

        texts = [build_text(item) for item in data]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        out_rows = []
        for item, emb in zip(data, embeddings):
            row = dict(item)
            row["embedding_model"] = EMBEDDING_MODEL
            row["embedding_dim"] = int(emb.shape[0])
            row["embedding"] = emb.tolist()
            out_rows.append(row)

        write_jsonl(out_path, out_rows)
        print(f"  Done: {len(out_rows)} documents")

    print("\n========================================")
    print("All files processed successfully.")
    print("Embeddings are normalized → use cosine similarity via inner product.")

if __name__ == "__main__":
    main()
