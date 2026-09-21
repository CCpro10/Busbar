"""Answer Typed Decision Bench items with Busbar and emit the benchmark's own result format.

The benchmark's items map onto Busbar's public contract without reshaping the decisions: `noul`
becomes a Boolean question, `choice` a Choice over the same criteria, and `score` a Score over the
same ordered levels. Option IDs, level order and question text come from the frozen item, so the
gold answer keeps its meaning and the run stays scoreable by the upstream metric.

One state is compiled once per item and every question for that item is answered from that snapshot,
which is the intended usage and also what makes the run affordable on a laptop.

Items whose widest question exceeds the runtime's ceiling are recorded as skipped with a reason
rather than silently dropped, so coverage is visible in the report and in the scoring denominator.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from busbar.backends import load_backend
from busbar.runtime import Runtime
from busbar.schemas import MAX_ALTERNATIVES, ContextSpec, DecisionRequest


def read_items(path: Path) -> list[dict]:
    """Frozen items in file order, skipping ids-only rows whose text is not redistributable."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("state") and row.get("questions") and row.get("gold"):
            rows.append(row)
    return rows


def question_width(question: dict) -> int:
    """How many alternatives the question needs a label for."""
    if question["type"] == "noul":
        return 2
    return len(question.get("criteria") or {})


def too_wide(item: dict) -> str | None:
    """Report the first gold question the runtime cannot represent, if any."""
    for qid in item["gold"]:
        width = question_width(item["questions"][qid])
        if width > MAX_ALTERNATIVES:
            return f"{qid} needs {width} alternatives; ceiling is {MAX_ALTERNATIVES}"
    return None


def as_busbar_question(question: dict) -> dict:
    """Translate one typed question, preserving option identity and level order."""
    kind = question["type"]
    instructions = question["instructions"]
    if kind == "noul":
        return {"type": "boolean", "question": instructions}
    criteria = question["criteria"]
    if kind == "choice":
        return {
            "type": "choice",
            "question": instructions,
            "options": [{"id": key, "description": criteria[key]} for key in criteria],
        }
    # Score levels are an ordered list; the level's position is its numeric value, matching the
    # benchmark's integer gold labels.
    return {
        "type": "score",
        "question": instructions,
        "levels": [
            {"value": float(index), "description": text} for index, text in enumerate(criteria)
        ],
    }


def to_answer(question: dict, decision) -> dict:
    """Emit the upstream answer shape: a probability for noul, a distribution otherwise."""
    kind = question["type"]
    if kind == "noul":
        # Busbar's Boolean options are ordered (true, false); the benchmark wants P(yes).
        return {"type": "noul", "noul": decision.probabilities[0]}
    probabilities = dict(zip(decision.option_ids, decision.probabilities, strict=True))
    if kind == "choice":
        return {
            "type": "choice",
            "choice": decision.selected,
            "probabilities": probabilities,
            "confidence": 1.0 - decision.normalized_entropy,
        }
    # Score answers are keyed by level index, matching how the upstream metric reads them.
    levels = {str(int(float(key))): value for key, value in probabilities.items()}
    return {
        "type": "score",
        "score": float(decision.value),
        "probabilities": levels,
        "confidence": 1.0 - decision.normalized_entropy,
    }


def answer_item(runtime: Runtime, item: dict, batch: int) -> tuple[dict, float]:
    """Compile the item's state once, then answer its gold questions from that snapshot."""
    started = time.perf_counter()
    context = runtime.compile_context(
        ContextSpec(namespace="tdb", state=item["state"], instructions="")
    )
    answers: dict[str, dict] = {}
    ids = list(item["gold"])
    try:
        for start in range(0, len(ids), batch):
            part = ids[start : start + batch]
            result = runtime.decide(
                DecisionRequest(
                    namespace="tdb",
                    snapshot_id=context.snapshot.id,
                    questions={qid: as_busbar_question(item["questions"][qid]) for qid in part},
                )
            )
            for qid in part:
                answers[qid] = to_answer(item["questions"][qid], result.decisions[qid])
    finally:
        runtime.delete_context(context.snapshot.id, namespace="tdb")
    return answers, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True, help="frozen suites root")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--questions-per-request", type=int, default=16)
    parser.add_argument("--suite", action="append", default=None, help="limit to suites")
    parser.add_argument("--limit", type=int, default=0, help="first N items per suite; 0 = all")
    parser.add_argument("--label", default="busbar", help="system label in the result rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")

    directories = sorted((args.bench / "suites").iterdir())
    if args.suite:
        wanted = set(args.suite)
        directories = [d for d in directories if d.name in wanted]
    if not directories:
        parser.error("no matching suites")

    backend = load_backend(
        "mlx", args.model, args.revision, dtype=args.dtype, batch_size=args.batch_size
    )
    runtime = Runtime(backend)
    rows: list[dict] = []
    skipped = 0
    started = time.perf_counter()
    for directory in directories:
        items = read_items(directory / "items.jsonl")
        if args.limit:
            items = items[: args.limit]
        for index, item in enumerate(items, 1):
            reason = too_wide(item)
            if reason:
                skipped += 1
                rows.append(
                    {
                        "id": item["id"],
                        "system": args.label,
                        "task": (item.get("meta") or {}).get("task"),
                        "status": 422,
                        "answers": "",
                        "skip_reason": reason,
                        "model": args.model,
                    }
                )
                continue
            answers, elapsed = answer_item(runtime, item, args.questions_per_request)
            rows.append(
                {
                    "id": item["id"],
                    "system": args.label,
                    "task": (item.get("meta") or {}).get("task"),
                    "status": 200,
                    "answers": json.dumps(answers, ensure_ascii=False),
                    "latency_ms": round(elapsed, 1),
                    "model": args.model,
                }
            )
            if index % args.progress_every == 0:
                print(f"{directory.name}: {index}/{len(items)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = args.output.with_suffix(".manifest.json")
    with manifest.open("x") as stream:
        json.dump(
            {
                "schema_version": 1,
                "label": args.label,
                "model": backend.identity,
                "batch_size": args.batch_size,
                "questions_per_request": args.questions_per_request,
                "alternative_ceiling": MAX_ALTERNATIVES,
                "suites": [d.name for d in directories],
                "rows": len(rows),
                "answered": len(rows) - skipped,
                "skipped_too_wide": skipped,
                "wall_seconds": round(time.perf_counter() - started, 2),
                "result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(f"answered {len(rows) - skipped}/{len(rows)} items; skipped {skipped} as too wide")


if __name__ == "__main__":
    main()
