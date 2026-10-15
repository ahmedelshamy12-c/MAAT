import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text):
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunk_lookup(chunks_path="chunks.json"):
    chunks = load_json(ROOT / chunks_path)
    by_id = {}
    by_doc_law = defaultdict(list)
    law_names = Counter()
    for idx, chunk in enumerate(chunks):
        chunk_id = f"chunk_{idx}"
        text = normalize_text(chunk["text"])
        law_name = chunk["law_name"]
        record = {
            "id": chunk_id,
            "text": chunk["text"],
            "law_name": law_name,
            "categories": chunk.get("categories", ""),
        }
        by_id[chunk_id] = record
        by_doc_law[(law_name, text)].append(chunk_id)
        law_names[law_name] += 1
    return chunks, by_id, by_doc_law, law_names


def chunk_ids_for_docs(docs, metas, by_doc_law):
    ids = []
    for doc, meta in zip(docs or [], metas or []):
        law_name = meta.get("law_name") if meta else None
        candidates = by_doc_law.get((law_name, normalize_text(doc)), [])
        ids.append(candidates[0] if candidates else None)
    return ids


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def group_means(records, key_field, metric_fields):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key_field]].append(record)
    output = {}
    for key, rows in grouped.items():
        output[key] = {"count": len(rows)}
        for metric in metric_fields:
            output[key][metric] = mean([row.get(metric) for row in rows])
    return output


def looks_like_abstention(answer):
    answer = normalize_text(answer).lower()
    if not answer:
        return False
    phrases = [
        "لا توجد",
        "لا يوجد",
        "لم أجد",
        "لم اجد",
        "غير موجود",
        "غير متاح",
        "لا تتضمن",
        "لا يحتوي",
        "لا أستطيع",
        "لا استطيع",
        "ليس في السياق",
        "غير مذكور",
        "لا يمكنني",
        "لا يمكن",
    ]
    return any(phrase in answer for phrase in phrases)
