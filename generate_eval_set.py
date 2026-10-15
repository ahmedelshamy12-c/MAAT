import ast
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import Dataset


ROOT = Path(__file__).resolve().parent
DATASET_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "datasets"
    / "dataflare___egypt-legal-corpus"
    / "default"
    / "0.0.0"
    / "4855942706f8b2bea152cbb8db8040fd8da83a14"
    / "egypt-legal-corpus-train.arrow"
)


def load_target_categories():
    tree = ast.parse((ROOT / "data.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TARGET_CATEGORIES":
                    return {elt.value for elt in node.value.elts}
    raise RuntimeError("TARGET_CATEGORIES not found in data.py")


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def find_article_starts(text):
    starts = []
    for marker in ("مادة", "ماده"):
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos == -1:
                break
            if pos >= 2 and text[pos - 2 : pos] == "ال":
                start = pos + len(marker)
                continue
            if pos > 0 and re.match(r"[\u0621-\u064A]", text[pos - 1]):
                start = pos + len(marker)
                continue
            tail = text[pos + len(marker) : pos + len(marker) + 20]
            match = re.search(r"[ \[:\(]*([0-9٠-٩]+)", tail)
            if match:
                starts.append((pos, match.group(1)))
            start = pos + len(marker)
    return sorted(set(starts))


def extract_articles(text, max_chars=950):
    text = clean(text)
    starts = find_article_starts(text)
    articles = []
    for i, (pos, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else min(len(text), pos + max_chars)
        body = clean(text[pos:end])
        if 120 <= len(body) <= 2500:
            articles.append({"number": number, "label": "المادة", "text": body[:max_chars].rstrip(" ،.")})
    if articles:
        return articles

    starts = []
    marker = "فقرة رقم"
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos == -1:
            break
        tail = text[pos + len(marker) : pos + len(marker) + 20]
        match = re.search(r"[ \[:\(]*([0-9٠-٩]+)", tail)
        if match:
            starts.append((pos, match.group(1)))
        start = pos + len(marker)
    for i, (pos, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else min(len(text), pos + max_chars)
        body = clean(text[pos:end])
        if 120 <= len(body) <= 2500:
            articles.append({"number": number, "label": "الفقرة", "text": body[:max_chars].rstrip(" ،.")})
    return articles


def topic_from_article(article_text):
    body = re.sub(r"^ماد[ةه]\s*[\[:\(]*\s*[0-9٠-٩]+\s*[\]\):\-]*", "", article_text)
    body = re.sub(r"^فقرة رقم\s*[\[:\(]*\s*[0-9٠-٩]+\s*[\]\):\-]*", "", body)
    body = clean(body)
    words = body.split()
    return " ".join(words[:8])


def answer_from_article(article_text):
    body = re.sub(r"^ماد[ةه]\s*[\[:\(]*\s*[0-9٠-٩]+\s*[\]\):\-]*", "", article_text)
    body = re.sub(r"^فقرة رقم\s*[\[:\(]*\s*[0-9٠-٩]+\s*[\]\):\-]*", "", body)
    body = clean(body)
    return "يقرر النص أن " + body[:850].rstrip(" ،.") + "."


def load_chunks():
    with (ROOT / "chunks.json").open(encoding="utf-8") as f:
        chunks = json.load(f)
    by_law = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        by_law[chunk["law_name"]].append((f"chunk_{idx}", clean(chunk["text"])))
    return by_law


def evidence_phrases(article_text):
    compact = clean(article_text)
    words = compact.split()
    phrases = []
    for start in (0, 8, 16, 28):
        part = " ".join(words[start : start + 14])
        if len(part) >= 45:
            phrases.append(part)
    return phrases


def source_chunk_ids(by_law, law_name, *article_texts):
    ids = []
    for article_text in article_texts:
        for phrase in evidence_phrases(article_text):
            for chunk_id, chunk_text in by_law.get(law_name, []):
                if phrase in chunk_text and chunk_id not in ids:
                    ids.append(chunk_id)
            if ids:
                break
    return ids


def row_category(row, targets):
    for category in row["categories"]:
        if category in targets:
            return category
    return None


def make_single(item_id, row, category, article, by_law, difficulty):
    law_name = row["law_name"]
    ids = source_chunk_ids(by_law, law_name, article["text"])
    if not ids:
        return None
    topic = topic_from_article(article["text"])
    return {
        "id": item_id,
        "question": f"ما الحكم الذي تقرره {article['label']} {article['number']} من {law_name} بشأن {topic}؟",
        "category": category,
        "expected_law_name": law_name,
        "reference_answer": answer_from_article(article["text"]),
        "question_type": "single_row",
        "difficulty": difficulty,
        "source_row_text": row["text"],
        "source_chunk_ids": ids[:4],
        "expected_abstain": False,
    }


def make_multi(item_id, row_a, row_b, category, article_a, article_b, by_law, difficulty):
    law_name = row_a["law_name"]
    ids = source_chunk_ids(by_law, law_name, article_a["text"], article_b["text"])
    if not ids:
        return None
    return {
        "id": item_id,
        "question": (
            f"في {law_name}، ما الحكم الوارد في {article_a['label']} {article_a['number']}، "
            f"وما الحكم الوارد في {article_b['label']} {article_b['number']}؟"
        ),
        "category": category,
        "expected_law_name": law_name,
        "reference_answer": (
            f"بالنسبة إلى {article_a['label']} {article_a['number']}: {answer_from_article(article_a['text'])} "
            f"وبالنسبة إلى {article_b['label']} {article_b['number']}: {answer_from_article(article_b['text'])}"
        ),
        "question_type": "multi_row",
        "difficulty": difficulty,
        "source_row_text": "ROW_A:\n" + row_a["text"] + "\n\nROW_B:\n" + row_b["text"],
        "source_chunk_ids": ids[:6],
        "expected_abstain": False,
    }


TRAPS = [
    {
        "question": "ما الحد الأقصى لمدة الحبس الاحتياطي في مخالفات البناء؟",
        "category": "الأجراءات الجنائية",
        "reference_answer": "لا توجد في النصوص محل التقييم إجابة كافية عن مدة الحبس الاحتياطي في مخالفات البناء.",
        "difficulty": "medium",
    },
    {
        "question": "هل يجيز قانون المواريث المصري توريث الحيوان الأليف إذا أوصى له المتوفى بجزء من التركة؟",
        "category": "قوانين الأحوال الشخصية",
        "reference_answer": "لا توجد في النصوص محل التقييم إجابة تثبت جواز توريث الحيوان الأليف أو معاملته كوريث.",
        "difficulty": "easy",
    },
    {
        "question": "ما نسبة الضريبة الجمركية المقررة على العملات الرقمية عند إدخالها إلى البلاد؟",
        "category": "جمارك",
        "reference_answer": "لا تتضمن النصوص محل التقييم حكماً عن العملات الرقمية أو نسبة ضريبة جمركية عليها.",
        "difficulty": "medium",
    },
    {
        "question": "هل ينظم قانون التحكيم إجراءات التحكيم في منازعات عقود الفضاء التجاري؟",
        "category": "القوانين والتشريعات",
        "reference_answer": "لا توجد في النصوص محل التقييم إجابة عن منازعات عقود الفضاء التجاري.",
        "difficulty": "hard",
    },
    {
        "question": "ما العقوبة المقررة لاختراق حساب على منصة تواصل اجتماعي؟",
        "category": "العقوبات",
        "reference_answer": "لا تتضمن النصوص محل التقييم حكماً محدداً عن اختراق حسابات منصات التواصل الاجتماعي.",
        "difficulty": "medium",
    },
    {
        "question": "هل يقرر الدستور مدة محددة لإجازة الأبوة للموظف العام؟",
        "category": "الدستور",
        "reference_answer": "لا توجد في النصوص محل التقييم إجابة عن مدة إجازة الأبوة للموظف العام.",
        "difficulty": "easy",
    },
    {
        "question": "ما ميعاد الطعن الإلكتروني على الأحكام عبر بوابة رقمية رسمية؟",
        "category": "المرافعات",
        "reference_answer": "لا تتضمن النصوص محل التقييم حكماً عن الطعن الإلكتروني عبر بوابة رقمية رسمية.",
        "difficulty": "medium",
    },
    {
        "question": "هل تسقط النفقة الزوجية إذا تأخرت الزوجة عن الرد على رسالة إلكترونية من الزوج؟",
        "category": "قوانين الأحوال الشخصية",
        "reference_answer": "لا يوجد في النصوص محل التقييم حكم يربط سقوط النفقة الزوجية بعدم الرد على رسالة إلكترونية.",
        "difficulty": "hard",
    },
    {
        "question": "ما إجراءات الإفراج الجمركي عن مركبات ذاتية القيادة تعمل دون سائق؟",
        "category": "جمارك",
        "reference_answer": "لا تتضمن النصوص محل التقييم إجراءات خاصة بالإفراج الجمركي عن مركبات ذاتية القيادة.",
        "difficulty": "medium",
    },
]


def main():
    targets = load_target_categories()
    ds = Dataset.from_file(str(DATASET_CACHE))
    by_law_chunks = load_chunks()

    rows_by_category = defaultdict(list)
    rows_by_law = defaultdict(list)
    for idx, row in enumerate(ds):
        category = row_category(row, targets)
        if not category:
            continue
        if "شرح" in row["law_name"]:
            continue
        row = dict(row)
        row["_row_index"] = idx
        row["_category"] = category
        row["_articles"] = extract_articles(row["text"])
        if row["_articles"]:
            rows_by_category[category].append(row)
            rows_by_law[row["law_name"]].append(row)

    items = []
    counters = defaultdict(int)
    desired_per_category = 10
    difficulties = ["easy", "medium", "hard", "medium"]

    for category in sorted(targets):
        article_cursor = 0
        rows = sorted(rows_by_category[category], key=lambda r: (-len(r["_articles"]), r["_row_index"]))
        while counters[category] < desired_per_category and rows:
            row = rows[article_cursor % len(rows)]
            article_index = (article_cursor // len(rows)) * 2
            if article_index >= len(row["_articles"]):
                article_cursor += 1
                if article_cursor > len(rows) * 80:
                    break
                continue
            item = make_single(
                f"eval_{len(items)+1:03d}",
                row,
                category,
                row["_articles"][article_index],
                by_law_chunks,
                difficulties[len(items) % len(difficulties)],
            )
            article_cursor += 1
            if item:
                items.append(item)
                counters[category] += 1

    multi_needed = 12
    multi_by_category = defaultdict(int)
    for law_name, law_rows in sorted(rows_by_law.items(), key=lambda kv: -len(kv[1])):
        if len(law_rows) < 2:
            continue
        row_a, row_b = law_rows[0], law_rows[1]
        category = row_a["_category"]
        if multi_by_category[category] >= 3:
            continue
        max_pairs = min(3, len(row_a["_articles"]) // 2, len(row_b["_articles"]) // 2)
        for pair_index in range(max_pairs):
            if sum(1 for item in items if item["question_type"] == "multi_row") >= multi_needed:
                break
            if multi_by_category[category] >= 3:
                break
            article_a = row_a["_articles"][pair_index * 2]
            article_b = row_b["_articles"][pair_index * 2 + 1]
            item = make_multi(
                f"eval_{len(items)+1:03d}",
                row_a,
                row_b,
                category,
                article_a,
                article_b,
                by_law_chunks,
                "hard" if pair_index % 2 else "medium",
            )
            if item:
                items.append(item)
                multi_by_category[category] += 1
        if sum(1 for item in items if item["question_type"] == "multi_row") >= multi_needed:
            break

    for trap in TRAPS:
        items.append(
            {
                "id": f"eval_{len(items)+1:03d}",
                "question": trap["question"],
                "category": trap["category"],
                "expected_law_name": None,
                "reference_answer": trap["reference_answer"],
                "question_type": "trap",
                "difficulty": trap["difficulty"],
                "source_row_text": "",
                "source_chunk_ids": [],
                "expected_abstain": True,
            }
        )

    for i, item in enumerate(items, 1):
        item["id"] = f"eval_{i:03d}"

    output = ROOT / "eval_set.json"
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = defaultdict(int)
    for item in items:
        summary[(item["category"], item["question_type"])] += 1
    print(f"wrote {len(items)} items to {output}")
    for key in sorted(summary):
        print(key, summary[key])


if __name__ == "__main__":
    main()
