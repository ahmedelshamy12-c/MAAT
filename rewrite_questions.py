import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def get_models():
    configured = os.getenv("GROQ_CHAT_MODELS")
    if not configured:
        return DEFAULT_MODELS
    models = [model.strip() for model in configured.split(",") if model.strip()]
    return models or DEFAULT_MODELS


def load_chunks():
    path = ROOT / "chunks.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    return {f"chunk_{idx}": chunk["text"] for idx, chunk in enumerate(chunks)}


def context_for_item(item, chunk_texts):
    chunk_context = "\n\n".join(
        clean(chunk_texts.get(chunk_id, "")) for chunk_id in item.get("source_chunk_ids", [])
    )
    if chunk_context:
        return chunk_context[:2200]

    # Fallback for unusual cases: source rows can be very large, so keep enough
    # local context for wording without sending whole laws.
    return clean(item.get("source_row_text", ""))[:2200]


def extract_question(text):
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("question"), str):
            return clean(data["question"])
    except json.JSONDecodeError:
        pass

    match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
    if match:
        return clean(match.group(1))
    return clean(text.strip("`"))


BAD_TERMS = [
    "المجرح",
    "المجروح",
    "المادة الثالثة",
    "وفقًا للمادة",
    "وفقاً للمادة",
    "من المادة",
]


def validate_question(question):
    if len(question) < 20:
        return "Question too short"
    if any(term in question for term in BAD_TERMS):
        return "Question contains awkward or explicitly article-shaped wording"
    if question.count("؟") > 1:
        return "Question contains multiple question marks"
    template_bits = [
        "ما الحكم الذي تقرره",
        "ما الحكم الوارد",
        "ماذا تنص المادة",
        "ماذا تقرر المادة",
    ]
    if any(bit in question for bit in template_bits):
        return "Question still uses a template pattern"
    return None


def rewrite_question(client, item, context, models, attempt=1, previous_error=None):
    retry_note = ""
    if previous_error:
        retry_note = f"""

تنبيه مهم: الصياغة السابقة رُفضت للسبب الآتي: {previous_error}
أعد الصياغة بشكل طبيعي أكثر، ولا تذكر رقم المادة أو الفقرة.
"""

    prompt = f"""أعد صياغة السؤال القانوني فقط باللغة العربية الفصحى المعاصرة.

المطلوب:
- اكتب سؤالاً طبيعياً قد يسأله شخص حقيقي.
- يجب أن تكون الإجابة المرجعية أدناه إجابة صحيحة ومباشرة عن السؤال.
- لا تذكر رقم المادة أو الفقرة صراحة. اسأل عن الواقعة أو القاعدة القانونية نفسها.
- نوّع الصياغة والبنية، ولا تستخدم قالباً مثل: "ما الحكم الذي تقرره...".
- لا تضف وقائع أو شروطاً غير موجودة في السياق.
- تجنّب الألفاظ الركيكة أو غير القانونية. استخدم مثلاً "المجني عليه" لا ألفاظاً غريبة.
- أعد JSON فقط بالشكل: {{"question": "..."}}
{retry_note}

السؤال الحالي:
{item["question"]}

التصنيف:
{item["category"]}

اسم القانون المتوقع:
{item["expected_law_name"]}

الصعوبة:
{item["difficulty"]}

الإجابة المرجعية التي يجب أن يجيب عنها السؤال:
{item["reference_answer"]}

السياق القانوني:
{context}
"""
    last_error = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
                max_completion_tokens=120,
                reasoning_effort="low",
            )
            question = extract_question(response.choices[0].message.content)
            if not question.endswith("؟"):
                question = question.rstrip(".،?") + "؟"
            validation_error = validate_question(question)
            if validation_error:
                if attempt < 2:
                    return rewrite_question(
                        client,
                        item,
                        context,
                        models,
                        attempt=attempt + 1,
                        previous_error=validation_error,
                    )
                raise ValueError(f"{validation_error}: {question!r}")
            return question
        except Exception as error:
            last_error = error
    raise RuntimeError("All configured Groq models failed") from last_error


def rewrite_items(input_path, output_path, limit=None, dry_run=False, sleep_seconds=0.0):
    load_dotenv()
    original_items = json.loads(input_path.read_text(encoding="utf-8"))
    original_by_id = {item["id"]: item for item in original_items}
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if not dry_run and partial_path.exists():
        items = json.loads(partial_path.read_text(encoding="utf-8"))
        print(f"resuming_from: {partial_path}", flush=True)
    else:
        items = original_items

    remaining_non_traps = [
        item
        for item in items
        if item.get("question_type") != "trap"
        and item["question"] == original_by_id[item["id"]]["question"]
    ]
    planned = len(remaining_non_traps) if limit is None else min(limit, len(remaining_non_traps))

    print(f"planned_llm_calls: {planned}")
    if len(remaining_non_traps) > 91:
        print(f"WARNING: full run would require {len(remaining_non_traps)} LLM calls, exceeding 91.")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=60.0)
    models = get_models()
    chunk_texts = load_chunks()

    rewritten = 0
    examples = []
    for item in items:
        if item.get("question_type") == "trap":
            continue
        if item["question"] != original_by_id[item["id"]]["question"]:
            continue
        if limit is not None and rewritten >= limit:
            break

        old_question = item["question"]
        context = context_for_item(item, chunk_texts)
        new_question = rewrite_question(client, item, context, models)
        item["question"] = new_question
        rewritten += 1
        print(f"[{rewritten}/{planned}] rewrote {item['id']}", flush=True)
        examples.append(
            {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "expected_law_name": item["expected_law_name"],
                "old_question": old_question,
                "new_question": new_question,
                "reference_answer": item["reference_answer"],
                "source_chunk_ids": item["source_chunk_ids"],
            }
        )
        if not dry_run:
            partial_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        if sleep_seconds:
            time.sleep(sleep_seconds)

    if not dry_run:
        output_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        if partial_path.exists():
            partial_path.unlink()
        print(f"wrote: {output_path}")

    if dry_run or limit is not None:
        print(json.dumps(examples, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_set.json")
    parser.add_argument("--output", default="eval_set_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    rewrite_items(
        ROOT / args.input,
        ROOT / args.output,
        limit=args.limit,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
