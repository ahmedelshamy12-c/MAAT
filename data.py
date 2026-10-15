from datasets import load_dataset
import tiktoken
import json
#download the dataset from huggingface
ds = load_dataset("dataflare/egypt-legal-corpus")["train"]

TARGET_CATEGORIES = {
    "القوانين والتشريعات",
    "جمارك",
    "المرافعات",
    "قوانين الأحوال الشخصية",
    "العقوبات",
    "الدستور",
    "الأجراءات الجنائية",
}

filtered = ds.filter(lambda row: any(c in TARGET_CATEGORIES for c in row["categories"]))



encoding = tiktoken.get_encoding("cl100k_base")



# Fixed Size Chunking with Overlap.
def chunk_text(text, chunk_size=500, overlap=80):
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(encoding.decode(tokens[start:end]))
        start += chunk_size - overlap
    return chunks

all_chunks = []
for row in filtered:
    for chunk in chunk_text(row["text"]):
        all_chunks.append({
            "text": chunk,
            "law_name": row["law_name"],
            "categories": ", ".join(row["categories"]),
        })


with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False)

