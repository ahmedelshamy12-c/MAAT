import argparse
from collections import defaultdict
from pathlib import Path

from eval_utils import group_means, load_chunk_lookup, load_json, mean, write_json


ROOT = Path(__file__).resolve().parent


def reciprocal_rank(expected_law_name, retrieved_law_names):
    if not expected_law_name:
        return None
    for index, law_name in enumerate(retrieved_law_names or [], 1):
        if law_name == expected_law_name:
            return 1 / index
    return 0.0


def score_record(record, law_counts):
    expected = record.get("expected_law_name")
    retrieved_laws = record.get("retrieved_law_names") or []
    retrieved_chunks = set(cid for cid in record.get("retrieved_chunk_ids") or [] if cid)
    source_chunks = set(record.get("source_chunk_ids") or [])

    rr = reciprocal_rank(expected, retrieved_laws)
    recall = None if expected is None else float(expected in retrieved_laws)
    source_hit = None if not source_chunks else float(bool(source_chunks & retrieved_chunks))

    chunking_loss = False
    if expected and law_counts.get(expected, 0) > 0 and source_chunks and expected in retrieved_laws:
        chunking_loss = not bool(source_chunks & retrieved_chunks)

    return {
        "id": record["id"],
        "category": record["category"],
        "difficulty": record["difficulty"],
        "question_type": record["question_type"],
        "expected_law_name": expected,
        "retrieved_law_names": retrieved_laws,
        "retrieved_chunk_ids": record.get("retrieved_chunk_ids") or [],
        "source_chunk_ids": record.get("source_chunk_ids") or [],
        "recall_at_k": recall,
        "mrr": rr,
        "source_chunk_hit": source_hit,
        "chunking_loss": chunking_loss,
        "error": record.get("error"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_results.json")
    parser.add_argument("--output", default="retrieval_scores.json")
    args = parser.parse_args()

    records = load_json(ROOT / args.results)
    _, _, _, law_counts = load_chunk_lookup()

    scored = [
        score_record(record, law_counts)
        for record in records
        if record.get("question_type") in {"single_row", "multi_row"}
    ]
    overall = {
        "count": len(scored),
        "recall_at_k": mean([row["recall_at_k"] for row in scored]),
        "mrr": mean([row["mrr"] for row in scored]),
        "source_chunk_hit": mean([row["source_chunk_hit"] for row in scored]),
        "chunking_loss_rate_when_law_retrieved": mean(
            [float(row["chunking_loss"]) for row in scored if row["expected_law_name"] in row["retrieved_law_names"]]
        ),
        "chunking_loss_count": sum(1 for row in scored if row["chunking_loss"]),
    }
    output = {
        "overall": overall,
        "by_category": group_means(scored, "category", ["recall_at_k", "mrr", "source_chunk_hit"]),
        "by_difficulty": group_means(scored, "difficulty", ["recall_at_k", "mrr", "source_chunk_hit"]),
        "items": scored,
    }
    write_json(ROOT / args.output, output)
    print("overall", overall)
    print(f"wrote {ROOT / args.output}")


if __name__ == "__main__":
    main()
