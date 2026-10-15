import argparse
import os
import time
from pathlib import Path

from eval_utils import chunk_ids_for_docs, load_chunk_lookup, load_json, write_json


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval_set_v2.json")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    # Keep RAG.py unmodified, but avoid stale/unavailable default fallbacks where possible.
    os.environ.setdefault("GROQ_CHAT_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b")

    from RAG import generate_answer, retrieve

    eval_items = load_json(ROOT / args.eval_set)
    if args.limit is not None:
        eval_items = eval_items[: args.limit]

    _, _, by_doc_law, _ = load_chunk_lookup()
    output_path = ROOT / args.output
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        results = load_json(partial_path)
        completed_ids = {record["id"] for record in results}
        print(f"resuming_from {partial_path} with {len(completed_ids)} completed", flush=True)
    else:
        results = []
        completed_ids = set()

    for index, item in enumerate(eval_items, 1):
        if item["id"] in completed_ids:
            continue
        print(f"[{index}/{len(eval_items)}] {item['id']}", flush=True)
        record = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "expected_law_name": item["expected_law_name"],
            "reference_answer": item["reference_answer"],
            "question_type": item["question_type"],
            "difficulty": item["difficulty"],
            "expected_abstain": item.get("expected_abstain", False),
            "source_chunk_ids": item.get("source_chunk_ids", []),
            "contexts": [],
            "retrieved_law_names": [],
            "retrieved_chunk_ids": [],
            "answer": None,
            "generation_metas": [],
            "error": None,
        }
        try:
            docs, metas = retrieve(item["question"], k=args.k)
            record["contexts"] = docs
            record["retrieved_law_names"] = [meta.get("law_name") for meta in metas]
            record["retrieved_chunk_ids"] = chunk_ids_for_docs(docs, metas, by_doc_law)
        except Exception as error:
            record["error"] = f"retrieve: {type(error).__name__}: {error}"

        try:
            answer, generation_metas = generate_answer(item["question"], k=args.k)
            record["answer"] = answer
            record["generation_metas"] = generation_metas
        except Exception as error:
            existing = record["error"]
            record["error"] = (existing + " | " if existing else "") + f"generate: {type(error).__name__}: {error}"

        results.append(record)
        write_json(partial_path, results)
        if args.sleep:
            time.sleep(args.sleep)

    write_json(output_path, results)
    if partial_path.exists():
        partial_path.unlink()
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
