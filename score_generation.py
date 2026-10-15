import argparse
import copy
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from eval_utils import group_means, load_json, looks_like_abstention, mean, write_json


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
RAGAS_TIMEOUT_SECONDS = 60
RAGAS_MAX_RETRIES = 2
RAGAS_MAX_WAIT_SECONDS = 5
DEFAULT_RAGAS_LIMIT = 20
DEFAULT_RAGAS_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]
DEFAULT_RAGAS_MAX_TOKENS = 4096
DEFAULT_RAGAS_METRIC_SLEEP_SECONDS = 20
DEFAULT_RAGAS_METRIC_RETRIES = 2
DEFAULT_RAGAS_RETRY_SLEEP_SECONDS = 20


def patch_ragas_vertexai_import():
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class ChatVertexAI:
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def build_ragas_judge(model_name):
    patch_ragas_vertexai_import()
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    class GroqRagasLLMWrapper(LangchainLLMWrapper):
        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return super().generate_text(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks)

        async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return await super().agenerate_text(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq-backed RAGAS judging")
    chat_llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0,
        timeout=RAGAS_TIMEOUT_SECONDS,
        max_retries=0,
        max_tokens=int(os.getenv("RAGAS_MAX_TOKENS", DEFAULT_RAGAS_MAX_TOKENS)),
        n=1,
    )
    return GroqRagasLLMWrapper(
        chat_llm,
        run_config=RunConfig(
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,
            max_wait=RAGAS_MAX_WAIT_SECONDS,
            max_workers=1,
        ),
    )


def default_ragas_model():
    return os.getenv("RAGAS_GROQ_MODEL", ",".join(DEFAULT_RAGAS_MODELS))


def parse_model_candidates(model_arg):
    models = [model.strip() for model in (model_arg or "").split(",") if model.strip()]
    return models or DEFAULT_RAGAS_MODELS


def validate_groq_model(model_name):
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq-backed RAGAS judging")
    models = Groq(api_key=api_key).models.list().data
    available_model_ids = {model.id for model in models}
    if model_name not in available_model_ids:
        raise RuntimeError(f"Groq model is not available to this key: {model_name}")


def build_embeddings():
    patch_ragas_vertexai_import()
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAGAS_EMBED_MODEL", "intfloat/multilingual-e5-small"),
        encode_kwargs={"normalize_embeddings": True},
    )


def format_error(error):
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def run_ragas(non_trap_records, model_names, batch_size):
    patch_ragas_vertexai_import()
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    eligible_records = [record for record in non_trap_records if not record.get("error") and record.get("answer")]
    if not eligible_records:
        return [], {"error": "No non-trap rows with answers available for RAGAS"}

    model_errors = []
    judges = {}
    for model_name in model_names:
        try:
            validate_groq_model(model_name)
            judges[model_name] = build_ragas_judge(model_name)
        except Exception as error:
            error_info = format_error(error)
            model_errors.append({"model": model_name, **error_info})

    if not judges:
        return [], {
            "error": "No configured Groq RAGAS judge model is available",
            "model_errors": model_errors,
        }
    selected_model = next(iter(judges))

    embeddings = LangchainEmbeddingsWrapper(build_embeddings())
    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT_SECONDS,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT_SECONDS,
        max_workers=1,
    )
    metric_templates = [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ]
    rows = []
    metric_errors = []
    metric_sleep_seconds = float(os.getenv("RAGAS_METRIC_SLEEP_SECONDS", DEFAULT_RAGAS_METRIC_SLEEP_SECONDS))
    metric_retries = int(os.getenv("RAGAS_METRIC_RETRIES", DEFAULT_RAGAS_METRIC_RETRIES))
    retry_sleep_seconds = float(os.getenv("RAGAS_RETRY_SLEEP_SECONDS", DEFAULT_RAGAS_RETRY_SLEEP_SECONDS))
    first_metric_call = True
    for record in eligible_records:
        ragas_row = {
            "user_input": record["question"],
            "response": record.get("answer") or "",
            "retrieved_contexts": record.get("contexts") or [],
            "reference": record.get("reference_answer") or "",
        }
        scored = {}
        item_errors = {}
        for metric_name, metric_template in metric_templates:
            if not first_metric_call:
                time.sleep(metric_sleep_seconds)
            first_metric_call = False
            last_error_info = None
            for model_name, judge in judges.items():
                for attempt in range(1, metric_retries + 1):
                    if model_name != selected_model or attempt > 1:
                        time.sleep(retry_sleep_seconds)
                    metric = copy.deepcopy(metric_template)
                    if hasattr(metric, "llm"):
                        metric.llm = judge
                    if hasattr(metric, "embeddings"):
                        metric.embeddings = embeddings
                    try:
                        result = evaluate(
                            Dataset.from_list([ragas_row]),
                            metrics=[metric],
                            llm=judge,
                            embeddings=embeddings,
                            raise_exceptions=True,
                            batch_size=batch_size,
                            show_progress=True,
                            run_config=run_config,
                        )
                        value = result.to_pandas().to_dict(orient="records")[0].get(metric_name)
                        scored[metric_name] = None if value != value else value
                        last_error_info = None
                        break
                    except Exception as error:
                        last_error_info = {
                            "model": model_name,
                            "attempt": attempt,
                            **format_error(error),
                        }
                if metric_name in scored:
                    break
            if last_error_info:
                scored[metric_name] = None
                item_errors[metric_name] = last_error_info
                metric_errors.append(
                    {
                        "id": record["id"],
                        "metric": metric_name,
                        **last_error_info,
                    }
                )
        if item_errors:
            scored["metric_errors"] = item_errors
        rows.append(scored)

    ragas_error = {}
    ragas_error = {"model": selected_model}
    if model_errors:
        ragas_error["model_errors"] = model_errors
    if metric_errors:
        ragas_error["metric_error_count"] = len(metric_errors)
        ragas_error["metric_errors"] = metric_errors
    return rows, ragas_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_results.json")
    parser.add_argument("--output", default="generation_scores.json")
    parser.add_argument(
        "--model",
        default=default_ragas_model(),
        help="Comma-separated Groq model IDs to try for RAGAS judging.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=int(os.getenv("RAGAS_LIMIT", DEFAULT_RAGAS_LIMIT)))
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    records = load_json(ROOT / args.results)
    traps = [record for record in records if record.get("question_type") == "trap"]
    non_traps = [record for record in records if record.get("question_type") != "trap"]
    if args.limit:
        non_traps = non_traps[: args.limit]

    abstention_items = []
    for record in traps:
        abstained = looks_like_abstention(record.get("answer"))
        abstention_items.append(
            {
                "id": record["id"],
                "category": record["category"],
                "difficulty": record["difficulty"],
                "answer": record.get("answer"),
                "abstained": abstained,
                "abstention_correct": float(abstained),
                "error": record.get("error"),
            }
        )

    ragas_rows = []
    ragas_error = {}
    if not args.skip_ragas:
        try:
            ragas_rows, ragas_error = run_ragas(non_traps, parse_model_candidates(args.model), args.batch_size)
        except Exception as error:
            ragas_error = {"error": f"{type(error).__name__}: {error}"}

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ragas_by_id = {}
    for row, record in zip(ragas_rows, [r for r in non_traps if not r.get("error") and r.get("answer")]):
        scored = {
            "id": record["id"],
            "category": record["category"],
            "difficulty": record["difficulty"],
            "question_type": record["question_type"],
        }
        for metric in metric_names:
            scored[metric] = row.get(metric)
        if row.get("metric_errors"):
            scored["metric_errors"] = row["metric_errors"]
        ragas_by_id[record["id"]] = scored

    ragas_items = list(ragas_by_id.values())
    overall = {
        "ragas_count": len(ragas_items),
        "trap_count": len(abstention_items),
        "abstention_accuracy": mean([item["abstention_correct"] for item in abstention_items]),
    }
    for metric in metric_names:
        overall[metric] = mean([item.get(metric) for item in ragas_items])

    output = {
        "overall": overall,
        "by_category": group_means(ragas_items, "category", metric_names),
        "by_difficulty": group_means(ragas_items, "difficulty", metric_names),
        "trap_abstention_by_category": group_means(abstention_items, "category", ["abstention_correct"]),
        "ragas_error": ragas_error,
        "ragas_items": ragas_items,
        "trap_items": abstention_items,
        "answer_error_count": sum(1 for record in records if record.get("error")),
        "answer_error_types": dict(Counter((record.get("error") or "").split(":")[0] for record in records if record.get("error"))),
    }
    write_json(ROOT / args.output, output)
    print("overall", overall)
    if ragas_error:
        print("ragas_error", ragas_error)
    print(f"wrote {ROOT / args.output}")


if __name__ == "__main__":
    main()
