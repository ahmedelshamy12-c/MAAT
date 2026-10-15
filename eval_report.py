import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from eval_utils import load_json, mean


ROOT = Path(__file__).resolve().parent
GENERATION_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def pct(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def esc(value):
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def worst_cases(results, retrieval_items, generation_items, limit=10):
    retrieval_by_id = {item["id"]: item for item in retrieval_items}
    generation_by_id = {item["id"]: item for item in generation_items}
    rows = []
    for record in results:
        rid = record["id"]
        ret = retrieval_by_id.get(rid, {})
        gen = generation_by_id.get(rid, {})
        if record.get("question_type") == "trap":
            continue
        score = 0
        score += 0 if ret.get("recall_at_k") else 3
        score += 0 if ret.get("source_chunk_hit") else 2
        for metric in GENERATION_METRICS:
            value = gen.get(metric)
            score += 0.5 if value is None else 1 - float(value)
        rows.append((score, record, ret, gen))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[:limit]


def build_markdown(results, retrieval, generation):
    lines = ["# Egyptian Legal RAG Evaluation Report", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append(table(
        ["Metric", "Value"],
        [
            ["Items evaluated", len(results)],
            ["Retrieval Recall@k", retrieval["overall"].get("recall_at_k")],
            ["Retrieval MRR", retrieval["overall"].get("mrr")],
            ["Source chunk hit", retrieval["overall"].get("source_chunk_hit")],
            ["Chunking-loss count", retrieval["overall"].get("chunking_loss_count")],
            ["RAGAS faithfulness", generation["overall"].get("faithfulness")],
            ["RAGAS answer_relevancy", generation["overall"].get("answer_relevancy")],
            ["RAGAS context_precision", generation["overall"].get("context_precision")],
            ["RAGAS context_recall", generation["overall"].get("context_recall")],
            ["Trap abstention accuracy", generation["overall"].get("abstention_accuracy")],
            ["Answer error count", generation.get("answer_error_count")],
        ],
    ))
    if generation.get("ragas_error"):
        lines.append("")
        lines.append(f"RAGAS note: `{generation['ragas_error']}`")

    lines.append("")
    lines.append("## Retrieval By Category")
    lines.append("")
    lines.append(table(
        ["Category", "Count", "Recall@k", "MRR", "Source chunk hit"],
        [
            [cat, row.get("count"), row.get("recall_at_k"), row.get("mrr"), row.get("source_chunk_hit")]
            for cat, row in sorted(retrieval["by_category"].items())
        ],
    ))

    lines.append("")
    lines.append("## Retrieval By Difficulty")
    lines.append("")
    lines.append(table(
        ["Difficulty", "Count", "Recall@k", "MRR", "Source chunk hit"],
        [
            [diff, row.get("count"), row.get("recall_at_k"), row.get("mrr"), row.get("source_chunk_hit")]
            for diff, row in sorted(retrieval["by_difficulty"].items())
        ],
    ))

    lines.append("")
    lines.append("## Generation By Category")
    lines.append("")
    lines.append(table(
        ["Category", "Count", "Faithfulness", "Answer relevancy", "Context precision", "Context recall"],
        [
            [
                cat,
                row.get("count"),
                row.get("faithfulness"),
                row.get("answer_relevancy"),
                row.get("context_precision"),
                row.get("context_recall"),
            ]
            for cat, row in sorted(generation["by_category"].items())
        ],
    ))

    lines.append("")
    lines.append("## Worst 10 Cases")
    lines.append("")
    for _, record, ret, gen in worst_cases(results, retrieval["items"], generation["ragas_items"]):
        lines.append(f"### {record['id']} ({record['category']} / {record['difficulty']})")
        lines.append("")
        lines.append(f"- Question: {record['question']}")
        lines.append(f"- Expected law: {record['expected_law_name']}")
        lines.append(f"- Retrieved laws: {', '.join(record.get('retrieved_law_names') or [])}")
        lines.append(f"- Recall@k: {fmt(ret.get('recall_at_k'))}; MRR: {fmt(ret.get('mrr'))}; source chunk hit: {fmt(ret.get('source_chunk_hit'))}")
        lines.append(
            "- RAGAS: "
            f"faithfulness={fmt(gen.get('faithfulness'))}, "
            f"answer_relevancy={fmt(gen.get('answer_relevancy'))}, "
            f"context_precision={fmt(gen.get('context_precision'))}, "
            f"context_recall={fmt(gen.get('context_recall'))}"
        )
        lines.append(f"- Answer: {record.get('answer')}")
        if record.get("error"):
            lines.append(f"- Error: {record['error']}")
        lines.append("")
    return "\n".join(lines)


def metric_class(value):
    if value is None:
        return "score na"
    if value >= 0.8:
        return "score high"
    if value >= 0.6:
        return "score mid"
    return "score low"


def group_category_rows(results, retrieval_items, generation_items):
    result_by_id = {row["id"]: row for row in results}
    grouped = defaultdict(lambda: {"retrieval": [], "generation": []})

    for item in retrieval_items:
        result = result_by_id.get(item["id"], {})
        category = result.get("category") or item.get("category") or "Unknown"
        grouped[category]["retrieval"].append(item)

    for item in generation_items:
        result = result_by_id.get(item["id"], {})
        category = result.get("category") or item.get("category") or "Unknown"
        grouped[category]["generation"].append(item)

    rows = []
    for category, values in grouped.items():
        retrieval = values["retrieval"]
        generation = values["generation"]
        rows.append(
            {
                "category": category,
                "count": len(retrieval),
                "recall_at_k": mean([row.get("recall_at_k") for row in retrieval]),
                "mrr": mean([row.get("mrr") for row in retrieval]),
                "source_chunk_hit": mean([row.get("source_chunk_hit") for row in retrieval]),
                "faithfulness": mean([row.get("faithfulness") for row in generation]),
                "answer_relevancy": mean([row.get("answer_relevancy") for row in generation]),
                "context_precision": mean([row.get("context_precision") for row in generation]),
                "context_recall": mean([row.get("context_recall") for row in generation]),
            }
        )
    return sorted(rows, key=lambda row: row["category"])


def metric_cell(value):
    sort_value = "" if value is None else f"{value:.12f}"
    return f'<td data-sort="{sort_value}"><span class="{metric_class(value)}">{fmt(value)}</span></td>'


def category_table(rows):
    headers = [
        ("category", "Category"),
        ("count", "Count"),
        ("recall_at_k", "Recall@k"),
        ("mrr", "MRR"),
        ("source_chunk_hit", "Source hit"),
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Relevancy"),
        ("context_precision", "Precision"),
        ("context_recall", "Context recall"),
    ]
    head = "".join(
        f'<th scope="col" data-key="{key}" tabindex="0">{label}<span class="sort-mark"></span></th>'
        for key, label in headers
    )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f'<td class="category" data-sort="{esc(row["category"])}">{esc(row["category"])}</td>'
            f'<td data-sort="{row["count"]}">{row["count"]}</td>'
            f'{metric_cell(row["recall_at_k"])}'
            f'{metric_cell(row["mrr"])}'
            f'{metric_cell(row["source_chunk_hit"])}'
            f'{metric_cell(row["faithfulness"])}'
            f'{metric_cell(row["answer_relevancy"])}'
            f'{metric_cell(row["context_precision"])}'
            f'{metric_cell(row["context_recall"])}'
            "</tr>"
        )
    return f"""
    <div class="table-wrap">
      <table id="categoryTable">
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    """


def case_details(cases):
    blocks = []
    for rank, (score, record, ret, gen) in enumerate(cases, 1):
        retrieved_laws = ", ".join(record.get("retrieved_law_names") or [])
        ragas = ", ".join(f"{name}={fmt(gen.get(name))}" for name in GENERATION_METRICS)
        blocks.append(
            f"""
            <details class="case">
              <summary>
                <span class="case-rank">{rank}</span>
                <span class="case-title">{esc(record["id"])} &middot; {esc(record["category"])} &middot; {esc(record["difficulty"])}</span>
                <span class="case-score">risk {score:.2f}</span>
              </summary>
              <div class="case-body">
                <dl>
                  <dt>Question</dt><dd>{esc(record.get("question"))}</dd>
                  <dt>Expected law</dt><dd>{esc(record.get("expected_law_name"))}</dd>
                  <dt>Retrieved laws</dt><dd>{esc(retrieved_laws)}</dd>
                  <dt>Retrieval</dt><dd>Recall@k {fmt(ret.get("recall_at_k"))}; MRR {fmt(ret.get("mrr"))}; source hit {fmt(ret.get("source_chunk_hit"))}</dd>
                  <dt>RAGAS</dt><dd>{esc(ragas)}</dd>
                  <dt>Answer</dt><dd class="answer">{esc(record.get("answer"))}</dd>
                </dl>
              </div>
            </details>
            """
        )
    return "\n".join(blocks)


def card(label, value, detail):
    return f"""
    <section class="summary-card">
      <div class="card-label">{esc(label)}</div>
      <div class="card-value">{esc(value)}</div>
      <div class="card-detail">{esc(detail)}</div>
    </section>
    """


def build_html(results, retrieval, generation):
    category_rows = group_category_rows(results, retrieval["items"], generation["ragas_items"])
    cases = worst_cases(results, retrieval["items"], generation["ragas_items"], limit=10)
    overall = retrieval["overall"]
    gen = generation["overall"]
    ragas_note = generation.get("ragas_error") or {}
    ragas_note_text = json.dumps(ragas_note, ensure_ascii=False) if ragas_note else "No RAGAS errors recorded."

    return f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Egyptian Legal RAG Dashboard</title>
  <style>
    :root {{ color-scheme: light; --bg: #f6f7f4; --panel: #ffffff; --ink: #1f2923; --muted: #66736d; --line: #dfe5df; --green-bg: #dff4e5; --green: #176b3a; --yellow-bg: #fff1c7; --yellow: #8a5d00; --red-bg: #ffe1dc; --red: #a93624; --accent: #245f73; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Tahoma, Arial, sans-serif; line-height: 1.6; }}
    .page {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 44px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 22px; }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 30px; line-height: 1.25; }}
    h2 {{ font-size: 19px; margin: 30px 0 12px; }}
    .subtitle, .meta, .card-label, .card-detail, .legend, .case-score, dt {{ color: var(--muted); }}
    .subtitle {{ margin-top: 6px; }}
    .meta {{ font-size: 14px; text-align: left; direction: ltr; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .summary-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; min-height: 126px; }}
    .card-label, .card-detail, .legend, .case-score {{ font-size: 13px; }}
    .card-value {{ font-size: 34px; font-weight: 700; margin: 10px 0 2px; direction: ltr; text-align: right; }}
    .table-wrap {{ overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 960px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: middle; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 13px; background: #fbfcfa; cursor: pointer; user-select: none; position: sticky; top: 0; }}
    th:hover {{ color: var(--accent); }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .category {{ font-weight: 700; min-width: 210px; }}
    .score {{ display: inline-flex; align-items: center; justify-content: center; min-width: 64px; padding: 3px 9px; border-radius: 999px; direction: ltr; font-weight: 700; font-size: 13px; }}
    .score.high {{ color: var(--green); background: var(--green-bg); }}
    .score.mid {{ color: var(--yellow); background: var(--yellow-bg); }}
    .score.low {{ color: var(--red); background: var(--red-bg); }}
    .score.na {{ color: var(--muted); background: #eef1ee; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 14px; }}
    .cases {{ display: grid; gap: 10px; }}
    .case {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .case summary {{ cursor: pointer; display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; padding: 13px 15px; }}
    .case-rank {{ width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; background: #e7eef0; color: var(--accent); font-weight: 700; direction: ltr; }}
    .case-title {{ font-weight: 700; }}
    .case-score {{ direction: ltr; }}
    .case-body {{ padding: 0 16px 16px; border-top: 1px solid var(--line); }}
    dl {{ display: grid; grid-template-columns: 132px minmax(0, 1fr); gap: 8px 14px; margin: 16px 0 0; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; min-width: 0; }}
    .answer {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    .note {{ background: #eef4f5; color: #315765; border: 1px solid #d2e2e5; border-radius: 8px; padding: 12px 14px; direction: ltr; text-align: left; overflow-wrap: anywhere; font-size: 13px; }}
    @media (max-width: 860px) {{ header {{ display: block; }} .meta {{ text-align: right; margin-top: 10px; }} .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} dl {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 560px) {{ .page {{ width: min(100% - 20px, 1180px); padding-top: 18px; }} h1 {{ font-size: 24px; }} .summary-grid {{ grid-template-columns: 1fr; }} .case summary {{ grid-template-columns: auto 1fr; }} .case-score {{ grid-column: 2; }} }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>Egyptian Legal RAG Dashboard</h1>
        <div class="subtitle">Retrieval, generation, abstention, and worst-case review in one view.</div>
      </div>
      <div class="meta">{len(results)} evaluated items &middot; {len(generation.get("ragas_items", []))} RAGAS rows</div>
    </header>
    <section class="summary-grid" aria-label="Summary">
      {card("Retrieval Recall@k", fmt(overall.get("recall_at_k")), f'MRR {fmt(overall.get("mrr"))} / source hit {fmt(overall.get("source_chunk_hit"))}')}
      {card("RAGAS Faithfulness", fmt(gen.get("faithfulness")), f'Relevancy {fmt(gen.get("answer_relevancy"))}')}
      {card("Trap Abstention", fmt(gen.get("abstention_accuracy")), f'{generation.get("trap_count", 0)} trap questions')}
      {card("Chunking Loss", str(overall.get("chunking_loss_count", 0)), f'Rate when law retrieved {pct(overall.get("chunking_loss_rate_when_law_retrieved"))}')}
    </section>
    <h2>Category Scores</h2>
    <div class="legend"><span class="score high">0.800+</span><span class="score mid">0.600-0.799</span><span class="score low">&lt;0.600</span><span class="score na">n/a</span></div>
    {category_table(category_rows)}
    <h2>Worst Cases</h2>
    <section class="cases">{case_details(cases)}</section>
    <h2>RAGAS Note</h2>
    <pre class="note">{esc(ragas_note_text)}</pre>
  </main>
  <script>
    const table = document.getElementById('categoryTable');
    let sortState = {{ index: 0, dir: 1 }};
    function sortTable(index) {{
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const current = sortState.index === index ? -sortState.dir : 1;
      sortState = {{ index, dir: current }};
      rows.sort((a, b) => {{
        const av = a.cells[index].dataset.sort || a.cells[index].innerText.trim();
        const bv = b.cells[index].dataset.sort || b.cells[index].innerText.trim();
        const an = Number(av);
        const bn = Number(bv);
        if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * current;
        return av.localeCompare(bv, 'ar') * current;
      }});
      rows.forEach(row => tbody.appendChild(row));
      table.querySelectorAll('th').forEach((th, i) => {{
        th.querySelector('.sort-mark').textContent = i === index ? (current === 1 ? ' asc' : ' desc') : '';
      }});
    }}
    table.querySelectorAll('th').forEach((th, index) => {{
      th.addEventListener('click', () => sortTable(index));
      th.addEventListener('keydown', event => {{
        if (event.key === 'Enter' || event.key === ' ') {{
          event.preventDefault();
          sortTable(index);
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_results.json")
    parser.add_argument("--retrieval", default="retrieval_scores.json")
    parser.add_argument("--generation", default="generation_scores.json")
    parser.add_argument("--output", default="eval_report.md")
    parser.add_argument("--html-output", default="eval_dashboard.html")
    args = parser.parse_args()

    results = load_json(ROOT / args.results)
    retrieval = load_json(ROOT / args.retrieval)
    generation = load_json(ROOT / args.generation)

    markdown_output = ROOT / args.output
    html_output = ROOT / args.html_output
    markdown_output.write_text(build_markdown(results, retrieval, generation), encoding="utf-8")
    html_output.write_text(build_html(results, retrieval, generation), encoding="utf-8")
    print(f"wrote {markdown_output}")
    print(f"wrote {html_output}")


if __name__ == "__main__":
    main()
