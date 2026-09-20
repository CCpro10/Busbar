"""Reproducible scoring of SemIf-compatible, provenance-preserving labeled JSONL."""

import hashlib
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .compiler import alternatives, canonical_json
from .runtime import Runtime
from .schemas import ChoiceQuestion, ContextSpec, DecisionRequest


def metrics(records: list[dict]) -> dict:
    """Score every row; probability metrics remain unavailable if any prediction failed."""
    if not records:
        raise ValueError("evaluation requires at least one labeled row")
    correct = sum(row.get("correct", False) for row in records)
    classes = {row["gold_id"] for row in records}
    recalls = [
        sum(row.get("correct", False) for row in records if row["gold_id"] == label)
        / sum(row["gold_id"] == label for row in records)
        for label in classes
    ]
    valid = [row for row in records if "probabilities" in row]
    result = {
        "rows": len(records),
        "answered": len(valid),
        "correct": correct,
        "accuracy": correct / len(records),
        "balanced_accuracy": statistics.mean(recalls),
        "coverage": len(valid) / len(records),
    }
    if len(valid) == len(records):
        result["nll"] = statistics.mean(
            -math.log(max(row["probabilities"][row["gold_index"]], 1e-12)) for row in valid
        )
        result["brier"] = statistics.mean(
            sum((p - (i == row["gold_index"])) ** 2 for i, p in enumerate(row["probabilities"]))
            for row in valid
        )
        bins = []
        for index in range(10):
            part = [row for row in valid if min(9, int(max(row["probabilities"]) * 10)) == index]
            if part:
                bins.append(
                    {
                        "lower": index / 10,
                        "upper": (index + 1) / 10,
                        "count": len(part),
                        "accuracy": statistics.mean(row["correct"] for row in part),
                        "mean_confidence": statistics.mean(
                            max(row["probabilities"]) for row in part
                        ),
                    }
                )
        result["reliability_bins"] = bins
        result["ece_10_equal_width"] = sum(
            item["count"] * abs(item["accuracy"] - item["mean_confidence"]) for item in bins
        ) / len(records)
    return result


def _evaluation_groups(backend, source, data_format, allow_training_data):
    """Normalize public datasets while retaining the exact context instructions and labels."""
    import json

    groups = defaultdict(list)
    if data_format == "native":
        from .datasets import read_dataset
        from .head_artifact import overlaps_provenance

        rows = read_dataset(source)
        if not allow_training_data:
            for split in getattr(backend, "training_splits", {}).values():
                if overlaps_provenance(rows, split):
                    raise ValueError(
                        "evaluation overlaps head training/validation data; "
                        "use held-out data or explicitly --allow-training-data"
                    )
        for item in rows:
            row = {
                "id": item.id,
                "family": item.question.type,
                "group_id": item.group_id,
                "label": item.gold_index,
                "options": [{"id": key} for key, _ in alternatives(item.question)],
            }
            groups[canonical_json(item.context.model_dump())].append((row, item.question))
        return groups
    if data_format != "semif" or allow_training_data:
        raise ValueError("expected semif or native format; training override is native-only")
    rows = [json.loads(line) for line in source.read_bytes().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("evaluation needs nonempty rows with unique IDs")
    for row in rows:
        question = ChoiceQuestion(type="choice", question=row["question"], options=row["options"])
        if (
            isinstance(row["label"], bool)
            or not isinstance(row["label"], int)
            or not 0 <= row["label"] < len(question.options)
        ):
            raise ValueError(f"invalid gold label: {row['id']}")
        groups[canonical_json(ContextSpec(state=row["state"]).model_dump())].append((row, question))
    return groups


def evaluate(
    backend,
    source: Path,
    mode="cached",
    projection="auto",
    *,
    data_format="semif",
    allow_training_data=False,
) -> dict:
    """Run all frozen rows, grouping identical states while keeping per-row gold provenance."""
    import json

    payload = source.read_bytes()
    groups = _evaluation_groups(backend, source, data_format, allow_training_data)
    runtime = Runtime(backend)
    # Warm an unrelated request; it is excluded from both accuracy and elapsed time.
    warm = runtime.compile_context(ContextSpec(state="The box is blue."))
    runtime.decide(
        DecisionRequest(
            snapshot_id=warm.snapshot.id,
            projection=projection,
            questions={"warm": {"type": "boolean", "question": "Is the box blue?"}},
        )
    )
    runtime.clear()
    results, timings = [], []
    started = time.perf_counter()
    for context_json, group in groups.items():
        mark = time.perf_counter()
        spec = ContextSpec.model_validate(json.loads(context_json))
        compiled = runtime.compile_context(spec)
        for start in range(0, len(group), 64):
            batch = group[start : start + 64]
            request = DecisionRequest(
                snapshot_id=compiled.snapshot.id,
                namespace=spec.namespace,
                questions={row["id"]: question for row, question in batch},
                mode=mode,
                projection=projection,
            )
            output = runtime.decide(request)
            timings.append(
                {
                    "rows": len(batch),
                    "compile_ms": compiled.elapsed_ms,
                    "inference_ms": output.inference_ms,
                    "computed_tokens": output.computed_tokens,
                    "reused_prefix_tokens": output.reused_prefix_tokens,
                    "backend_details": output.backend_details,
                }
            )
            for row, _ in batch:
                decision = output.decisions[row["id"]]
                gold_id = row["options"][row["label"]]["id"]
                results.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "group_id": row["group_id"],
                        "gold_index": row["label"],
                        "gold_id": gold_id,
                        "correct": decision.selected == gold_id,
                        "provenance": row.get("provenance", {}),
                        **decision.model_dump(),
                    }
                )
        runtime.delete_context(compiled.snapshot.id, spec.namespace)
        timings[-1]["state_wall_ms"] = (time.perf_counter() - mark) * 1000
    elapsed = time.perf_counter() - started
    families = defaultdict(list)
    for row in results:
        families[row["family"]].append(row)
    per_family = {name: metrics(part) for name, part in families.items()}
    summary = metrics(results)
    summary["mean_family_balanced_accuracy"] = statistics.mean(
        part["balanced_accuracy"] for part in per_family.values()
    )
    summary["wall_seconds"] = elapsed
    summary["decisions_per_second"] = len(results) / elapsed
    return {
        "schema_version": 1,
        "model": backend.identity,
        "input_name": source.name,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "data_format": data_format,
        "allow_training_data": allow_training_data,
        "mode": mode,
        "projection": projection,
        "summary": summary,
        "families": per_family,
        "timings": timings,
        "predictions": results,
        "limitations": [
            "Scores use this runtime's prompt, not the original SemIf scorer prompt.",
            "Vocabulary mode uses label-token scores; head mode uses learned candidate scores.",
            (
                "Training-data reuse explicitly allowed; this is not held-out evaluation."
                if allow_training_data
                else "Uncalibrated probabilities; no fitting or temperature tuning on these rows."
            ),
            "Quality is dataset-specific; preserve source provenance and label limitations.",
            "Runtime errors abort the run; partial success is never published as full coverage.",
            "Wall time includes per-state prefill and fanout, excludes model load and warm-up.",
        ],
    }
