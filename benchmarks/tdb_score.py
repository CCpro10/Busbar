"""Score Typed Decision Bench runs with the benchmark's own published metric.

The scoring functions are imported from the upstream `typed_decision_bench` package rather than
reimplemented, so a Busbar run and the published Jev run are compared on one definition of
DecisionScore. Pass `--source-root` to point at a clean upstream checkout.

Withheld items (empty `state`) are excluded by default: their text is not redistributable, so Busbar
cannot answer them, and scoring them as 0 would silently penalise Busbar against a Jev column that
did answer them.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def upstream_revision(root: Path) -> str:
    """Record the scorer checkout, refusing locally modified source."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("upstream scorer checkout must be clean")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_withheld(item: dict) -> bool:
    """Items shipped as ids-only carry an empty state, question set or gold."""
    return not item.get("state") or not item.get("questions") or not item.get("gold")


def load_items(bench_root: Path, keep_withheld: bool) -> dict[str, list[dict]]:
    """Frozen suites in the runner's native format, keyed by suite name."""
    suites = {}
    for directory in sorted((bench_root / "suites").iterdir()):
        items = read_jsonl(directory / "items.jsonl")
        if not keep_withheld:
            items = [item for item in items if not is_withheld(item)]
        if items:
            suites[directory.name] = items
    return suites


def load_flat_results(path: Path) -> dict[str, dict]:
    """The published flat result view: one row per item, `answers` as a JSON string."""
    rows = {}
    for row in read_jsonl(path):
        answers = row.get("answers")
        if isinstance(answers, str):
            answers = json.loads(answers) if answers else {}
        rows[row["id"]] = {**row, "answers": answers or {}}
    return rows


def evaluate(suites: dict[str, list[dict]], rows: dict[str, dict], site_data) -> dict:
    """Per-suite and overall DecisionScore/accuracy under the benchmark's own definition."""
    per_suite = {}
    all_scores: list[float] = []
    answered = correct = 0
    total = 0
    for name, items in suites.items():
        block = site_data.suite_result(items, rows)
        per_suite[name] = block
        total += block["items"]
        answered += block["answered"]
        correct += block["correct"]
        # Recover the per-item scores behind the suite block so the overall figure stays
        # item-weighted rather than an average of suite averages.
        for item in items:
            row = rows.get(item["id"])
            score = site_data.item_proper_score(item, (row or {}).get("answers"))
            all_scores.append(score if row and score is not None else 0.0)
    return {
        "items": total,
        "answered": answered,
        "correct": correct,
        "decision_score": round(100 * sum(all_scores) / total, 2) if total else 0.0,
        "accuracy": round(correct / answered, 4) if answered else None,
        "per_suite": per_suite,
    }


def per_task(suites: dict[str, list[dict]], rows: dict[str, dict], site_data) -> dict:
    """Task-level DecisionScore and accuracy, using the same per-item score."""
    buckets: dict[str, list[float]] = defaultdict(list)
    hits: dict[str, list[int]] = defaultdict(list)
    for items in suites.values():
        for item in items:
            task = (item.get("meta") or {}).get("task") or "unknown"
            row = rows.get(item["id"])
            score = site_data.item_proper_score(item, (row or {}).get("answers"))
            buckets[task].append(score if row and score is not None else 0.0)
            if row and score is not None:
                qid = site_data.primary_of(item)
                question = item["questions"][qid]
                answer = (row.get("answers") or {}).get(qid) or {}
                predicted = site_data.top_answer(question, answer)
                if predicted is not None:
                    hits[task].append(
                        int(predicted == site_data.gold_key(question, item["gold"][qid]))
                    )
    return {
        task: {
            "items": len(scores),
            "decision_score": round(100 * statistics.mean(scores), 2),
            "accuracy": round(statistics.mean(hits[task]), 4) if hits[task] else None,
        }
        for task, scores in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", type=Path, required=True, help="frozen suites root (contains suites/)"
    )
    parser.add_argument(
        "--source-root", type=Path, required=True, help="clean upstream jevfish checkout"
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="flat per-item result file; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep-withheld",
        action="store_true",
        help="score the ids-only items too; they count as unanswered zeros",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")

    root = args.source_root.resolve()
    revision = upstream_revision(root)
    sys.path.insert(0, str(root / "benchmark"))
    from typed_decision_bench import site_data

    suites = load_items(args.bench, args.keep_withheld)
    report = {
        "schema_version": 1,
        "upstream_revision": revision,
        "withheld_excluded": not args.keep_withheld,
        "suite_counts": {name: len(items) for name, items in suites.items()},
        "systems": {},
    }
    for entry in args.result:
        label, _, path = entry.partition("=")
        path = Path(path)
        rows = load_flat_results(path)
        report["systems"][label] = {
            "source_file": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows_in_file": len(rows),
            **evaluate(suites, rows, site_data),
            "per_task": per_task(suites, rows, site_data),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for label, block in report["systems"].items():
        print(
            f"{label}: DecisionScore {block['decision_score']} | "
            f"accuracy {block['accuracy']} | answered {block['answered']}/{block['items']}"
        )


if __name__ == "__main__":
    main()
