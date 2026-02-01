import json
import heapq
import textwrap
import time
import re
import string
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

RAG_EMBEDDINGS_DIR = Path(r"C:\Users\mywke\PythonProjects\isp\embeddings_rag")
HIER_DIR = Path(r"C:\Users\mywke\PythonProjects\isp\embeddings_hrag")
PARENTS_PATH = HIER_DIR / "parents.jsonl"
CHILDREN_PATH = HIER_DIR / "children.jsonl"
PROMPT_PATH = Path(r"C:\Users\mywke\PythonProjects\isp\prompt.txt")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LLM_DTYPE = torch.float16
LLM_DEVICE_MAP = "auto"

RAG_TOP_K = 3
HRAG_TOP_PARENTS = 4
HRAG_TOP_K = 3
MAX_CONTEXT_CHARS = 6000

MAX_NEW_TOKENS = 700
TEMPERATURE = 0.2
DO_SAMPLE = True

WRAP_WIDTH = 96
SHOW_FULL_PROMPT = False
PROMPT_PREVIEW_CHARS = 2500

ANSWER_WRAP = True
SHOW_CONTEXT_PREVIEW = False
CONTEXT_PREVIEW_CHARS = 900

SHOW_FILE = True

def hrule(title: Optional[str] = None, char: str = "=", width: int = WRAP_WIDTH) -> str:
    if not title:
        return char * width
    title = f" {title} "
    side = max(0, width - len(title))
    left = side // 2
    right = side - left
    return (char * left) + title + (char * right)

def wrap(text: str, width: int = WRAP_WIDTH) -> str:
    return "\n".join(textwrap.fill(line, width=width) if line.strip() else ""
                     for line in text.splitlines())

def truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "\n...[TRUNCATED]..."

def render_table(cols: List[str], rows: List[List[str]]) -> str:
    widths = [len(c) for c in cols]
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    header_line = " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(cols)))

    lines = [header_line, sep_line]
    for r in rows:
        lines.append(" | ".join(r[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(lines)

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s

# ======================================================
# IO
# ======================================================

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()

def format_prompt(template: str, context: str, question: str) -> str:
    if ("{context}" in template) or ("{question}" in template):
        return template.format(context=context, question=question)

    return (
        template
        + "\n\n---\n"
        + "CONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
        + "\n\nINSTRUCTIONS:\n"
        + "- Answer using ONLY the context.\n"
        + "- If the answer is not in the context, say you don't know.\n"
        + "- Cite which items you used (by DATE/TITLE) in 1-2 lines.\n"
    )

# ======================================================
# COMMON DOCUMENT TEXT
# ======================================================

def build_doc_text(doc: Dict[str, Any]) -> str:
    date = doc.get("date", "")
    title = doc.get("title", "")
    summary = doc.get("summary", "")
    return f"DATE: {date}\nTITLE: {title}\nSUMMARY: {summary}".strip()

# ======================================================
# RAG RETRIEVAL
# ======================================================

def get_all_embedding_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.jsonl"))

def rag_retrieve_topk_streaming(
    query_vec: np.ndarray,
    embedding_files: List[Path],
    top_k: int
) -> Tuple[List[Tuple[float, Dict[str, Any], Path]], int]:
    """
    Returns: (topk_results, scanned_docs_count)
    """
    heap: List[Tuple[float, Dict[str, Any], Path]] = []
    scanned = 0

    for fpath in embedding_files:
        for doc in iter_jsonl(fpath):
            scanned += 1
            emb = doc.get("embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32)
            score = float(np.dot(query_vec, v))
            if len(heap) < top_k:
                heapq.heappush(heap, (score, doc, fpath))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, doc, fpath))

    return sorted(heap, key=lambda x: x[0], reverse=True), scanned

def rag_build_context(docs_scored: List[Tuple[float, Dict[str, Any], Path]], max_chars: int) -> str:
    parts = []
    total = 0
    for rank, (score, doc, fpath) in enumerate(docs_scored, start=1):
        header = f"[RANK {rank} | score={score:.4f}]"
        if SHOW_FILE:
            header += f"\nFILE: {fpath}"
        block = header + "\n" + build_doc_text(doc)
        if total + len(block) + 2 > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts).strip()

def rag_topk_table_str(docs_scored: List[Tuple[float, Dict[str, Any], Path]]) -> str:
    cols = ["Rank", "Score", "Date", "Title"]
    if SHOW_FILE:
        cols.append("File")
    rows = []
    for i, (score, doc, fpath) in enumerate(docs_scored, start=1):
        date = str(doc.get("date", ""))
        title = str(doc.get("title", "")).replace("\n", " ").strip()
        if len(title) > 70:
            title = title[:67] + "..."
        row = [str(i), f"{score:.4f}", date, title]
        if SHOW_FILE:
            row.append(str(fpath))
        rows.append(row)

    return render_table(cols, rows)

# ======================================================
# HRAG RETRIEVAL
# ======================================================

def load_parents() -> List[Dict[str, Any]]:
    if not PARENTS_PATH.exists():
        raise FileNotFoundError(f"Missing parents index: {PARENTS_PATH}")
    return list(iter_jsonl(PARENTS_PATH))

def load_children() -> List[Dict[str, Any]]:
    if not CHILDREN_PATH.exists():
        raise FileNotFoundError(f"Missing children index: {CHILDREN_PATH}")
    return list(iter_jsonl(CHILDREN_PATH))

def hrag_topk_parents(query_vec: np.ndarray, parents: List[Dict[str, Any]], k: int) -> Tuple[List[Tuple[float, Dict[str, Any]]], int]:
    heap: List[Tuple[float, Dict[str, Any]]] = []
    scanned = 0
    for p in parents:
        scanned += 1
        v = np.asarray(p["embedding"], dtype=np.float32)
        s = float(np.dot(query_vec, v))
        if len(heap) < k:
            heapq.heappush(heap, (s, p))
        else:
            if s > heap[0][0]:
                heapq.heapreplace(heap, (s, p))
    return sorted(heap, key=lambda x: x[0], reverse=True), scanned

def hrag_topk_children_filtered(
    query_vec: np.ndarray,
    children: List[Dict[str, Any]],
    allowed_parent_ids: Set[str],
    k: int
) -> Tuple[List[Tuple[float, Dict[str, Any]]], int]:
    heap: List[Tuple[float, Dict[str, Any]]] = []
    scanned = 0
    for c in children:
        if c.get("parent_id") not in allowed_parent_ids:
            continue
        scanned += 1
        v = np.asarray(c["embedding"], dtype=np.float32)
        s = float(np.dot(query_vec, v))
        if len(heap) < k:
            heapq.heappush(heap, (s, c))
        else:
            if s > heap[0][0]:
                heapq.heapreplace(heap, (s, c))
    return sorted(heap, key=lambda x: x[0], reverse=True), scanned

def hrag_build_context(children_scored: List[Tuple[float, Dict[str, Any]]], max_chars: int) -> str:
    parts = []
    total = 0
    for rank, (score, c) in enumerate(children_scored, start=1):
        header = f"[RANK {rank} | score={score:.4f}]"
        if SHOW_FILE:
            header += f"\nFILE: {c.get('source_file','')}"
        block = (
            header
            + "\n"
            + f"DATE: {c.get('date','')}\n"
            + f"TITLE: {c.get('title','')}\n"
            + f"SUMMARY: {c.get('summary','')}\n"
            + f"CHUNK_TEXT: {c.get('chunk_text','')}"
        ).strip()

        if total + len(block) + 2 > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts).strip()

def hrag_parents_table_str(parents_scored: List[Tuple[float, Dict[str, Any]]]) -> str:
    cols = ["Rank", "Score", "parent_id", "Year", "Month", "Docs"]
    if SHOW_FILE:
        cols.append("File")
    rows = []
    for i, (s, p) in enumerate(parents_scored, start=1):
        row = [
            str(i), f"{s:.4f}",
            str(p.get("parent_id", "")),
            str(p.get("year", "")),
            str(p.get("month", "")),
            str(p.get("doc_count", "")),
        ]
        if SHOW_FILE:
            row.append(str(p.get("source_file", "")))
        rows.append(row)
    return render_table(cols, rows)

def hrag_children_table_str(children_scored: List[Tuple[float, Dict[str, Any]]]) -> str:
    cols = ["Rank", "Score", "Date", "Title"]
    if SHOW_FILE:
        cols.append("File")
    rows = []
    for i, (s, c) in enumerate(children_scored, start=1):
        date = str(c.get("date", ""))
        title = str(c.get("title", "")).replace("\n", " ").strip()
        if len(title) > 70:
            title = title[:67] + "..."
        row = [str(i), f"{s:.4f}", date, title]
        if SHOW_FILE:
            row.append(str(c.get("source_file", "")))
        rows.append(row)

    return render_table(cols, rows)

# ======================================================
# LLM GENERATION
# ======================================================

def generate_answer(tokenizer, model, prompt: str) -> Tuple[str, float]:
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    enc = tokenizer(chat_text, return_tensors="pt", truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = output_ids[0, prompt_len:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    t1 = time.perf_counter()
    return answer, (t1 - t0)

# ======================================================
# METRICS
# ======================================================

def answer_factuality_heuristic(answer: str, retrieved_titles: List[str], retrieved_dates: List[str]) -> str:
    a = answer.lower()
    if "i don't know" in a or "i do not know" in a:
        return "abstained (likely grounded)"
    for d in retrieved_dates:
        if d and d in answer:
            return "likely grounded (mentions retrieved date)"
    for t in retrieved_titles:
        if not t:
            continue
        key = t.strip()
        key_seg = key[:25].lower()
        if len(key_seg) >= 12 and key_seg in a:
            return "likely grounded (mentions retrieved title fragment)"
    return "risky (no explicit link to retrieved context)"

# ======================================================
# FINAL SUMMARY PRINTING
# ======================================================

def print_final_summary(
    question: str,
    rag_topk_str: str,
    hrag_parents_str: str,
    hrag_topk_str: str,
    rag_answer: str,
    hrag_answer: str,
    metrics_rows: List[List[str]]
):
    print("\n" + hrule("FINAL SUMMARY") + "\n")

    # QUESTION
    print(hrule("QUESTION", "-"))
    print(wrap(question))
    print()

    # TOP-K
    print(hrule("RAG: TOP-K", "-"))
    print(rag_topk_str)
    print()

    print(hrule("HRAG: PARENTS ", "-"))
    print(hrag_parents_str)
    print()

    print(hrule("HRAG: TOP-K ", "-"))
    print(hrag_topk_str)
    print()

    # ANSWERS
    print(hrule("RAG: ANSWER", "-"))
    print(wrap(rag_answer) if ANSWER_WRAP else rag_answer)
    print()

    print(hrule("HRAG: ANSWER", "-"))
    print(wrap(hrag_answer) if ANSWER_WRAP else hrag_answer)
    print()

    # METRICS
    print(hrule("METRICS", "-"))
    cols = ["Metric", "RAG", "HRAG"]
    print(render_table(cols, metrics_rows))
    print("\n" + hrule(width=WRAP_WIDTH) + "\n")






def main():
    prompt_template = load_prompt_template(PROMPT_PATH)
    print(f"Prompt loaded from: {PROMPT_PATH}")

    # Embedding model
    emb_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {emb_device}")
    emb_model = SentenceTransformer(EMBEDDING_MODEL, device=emb_device)

    # LLM
    print(f"Loading LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=LLM_DTYPE,
        device_map=LLM_DEVICE_MAP,
    )
    model.eval()

    # Load emb
    rag_files = get_all_embedding_files(RAG_EMBEDDINGS_DIR)
    if not rag_files:
        raise FileNotFoundError(f"No RAG embeddings found under: {RAG_EMBEDDINGS_DIR}")
    print(f"RAG embedded files found: {len(rag_files)}")

    if not PARENTS_PATH.exists() or not CHILDREN_PATH.exists():
        raise FileNotFoundError("HRAG indices not found. Generate them first (make_hier_embeddings.py).")

    parents = load_parents()
    children = load_children()
    print(f"HRAG parents : {len(parents)}")
    print(f"HRAG children: {len(children)}")

    print("\n" + hrule("COMPARE RAG vs HRAG READY") + "\n")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("Question> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        # Highlight query
        print("\n" + hrule("RUN START"))
        print(hrule("QUERY", "-"))
        print(wrap(question))

        # Query embedding
        q = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        # =======================
        # RAG PIPELINE
        # =======================
        t0 = time.perf_counter()
        rag_docs, rag_scanned = rag_retrieve_topk_streaming(q, rag_files, RAG_TOP_K)
        t1 = time.perf_counter()
        rag_retrieval_s = t1 - t0

        rag_context = rag_build_context(rag_docs, MAX_CONTEXT_CHARS)
        rag_prompt = format_prompt(prompt_template, rag_context, question)

        rag_answer, rag_gen_s = generate_answer(tokenizer, model, rag_prompt)
        rag_total_s = rag_retrieval_s + rag_gen_s

        rag_topk_str = rag_topk_table_str(rag_docs)

        rag_titles = [str(doc.get("title", "")) for _, doc, _ in rag_docs]
        rag_dates = [str(doc.get("date", "")) for _, doc, _ in rag_docs]

        # =======================
        # HRAG PIPELINE
        # =======================
        t2 = time.perf_counter()
        hrag_parents_scored, hrag_parents_scanned = hrag_topk_parents(q, parents, HRAG_TOP_PARENTS)
        allowed = {p.get("parent_id") for _, p in hrag_parents_scored}

        hrag_children_scored, hrag_children_scanned = hrag_topk_children_filtered(q, children, allowed, HRAG_TOP_K)
        t3 = time.perf_counter()
        hrag_retrieval_s = t3 - t2

        hrag_context = hrag_build_context(hrag_children_scored, MAX_CONTEXT_CHARS)
        hrag_prompt = format_prompt(prompt_template, hrag_context, question)

        hrag_answer, hrag_gen_s = generate_answer(tokenizer, model, hrag_prompt)
        hrag_total_s = hrag_retrieval_s + hrag_gen_s

        hrag_parents_str = hrag_parents_table_str(hrag_parents_scored)
        hrag_topk_str = hrag_children_table_str(hrag_children_scored)

        hrag_titles = [str(c.get("title", "")) for _, c in hrag_children_scored]
        hrag_dates = [str(c.get("date", "")) for _, c in hrag_children_scored]

        # =======================
        # METRICS
        # =======================

        rag_fact = answer_factuality_heuristic(rag_answer, rag_titles, rag_dates)
        hrag_fact = answer_factuality_heuristic(hrag_answer, hrag_titles, hrag_dates)

        hrag_scanned_total = hrag_parents_scanned + hrag_children_scanned
        if rag_scanned > 0:
            reduction = 1.0 - (hrag_scanned_total / float(rag_scanned))
        else:
            reduction = 0.0

        # metrics table
        def fmt_opt(x: Optional[float], digits=4) -> str:
            if x is None:
                return "N/A"
            return f"{x:.{digits}f}"

        metrics_rows = [
            ["Latency retrieval (s)", f"{rag_retrieval_s:.3f}", f"{hrag_retrieval_s:.3f}"],
            ["Latency generation (s)", f"{rag_gen_s:.3f}", f"{hrag_gen_s:.3f}"],
            ["Latency total (s)", f"{rag_total_s:.3f}", f"{hrag_total_s:.3f}"],
            ["Answer Factuality (heur.)", rag_fact, hrag_fact],
            ["Search Space Reduction", "baseline", f"{reduction:.1%}"],
            ["Scanned items", f"{rag_scanned}", f"{hrag_scanned_total} (parents+children)"],
        ]

        # =======================
        # FINAL OUTPUT
        # =======================
        print_final_summary(
            question=question,
            rag_topk_str=rag_topk_str,
            hrag_parents_str=hrag_parents_str,
            hrag_topk_str=hrag_topk_str,
            rag_answer=rag_answer if rag_answer else "[EMPTY OUTPUT]",
            hrag_answer=hrag_answer if hrag_answer else "[EMPTY OUTPUT]",
            metrics_rows=metrics_rows
        )


if __name__ == "__main__":
    main()
