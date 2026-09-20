"""Reproducible scoring of SemIf-compatible, provenance-preserving labeled JSONL."""

import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .compiler import alternatives, canonical_json
from .datasets import LabeledDecision, SemIfExample, parse_dataset
from .head_artifact import overlaps_provenance
from .runtime import Runtime
from .schemas import ContextSpec, DecisionRequest
from .storage import FileSnapshot


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


@dataclass(frozen=True)
class EvaluationInput:
    """Validated rows and provenance captured before the CLI loads a heavyweight backend."""

    source: FileSnapshot
    data_format: str
    rows: tuple[LabeledDecision | SemIfExample, ...]


def prepare_evaluation(source: Path, data_format="semif") -> EvaluationInput:
    """Reject every malformed row, including the last, without any model execution."""
    formats = {"native": LabeledDecision, "semif": SemIfExample}
    if data_format not in formats:
        raise ValueError("expected semif or native dataset format")
    snapshot = FileSnapshot.read(source)
    return EvaluationInput(
        snapshot, data_format, tuple(parse_dataset(snapshot, formats[data_format]))
    )


def _evaluation_groups(backend, inputs: EvaluationInput, allow_training_data):
    """Map validated records to runtime questions after checking native training leakage."""
    if allow_training_data and inputs.data_format != "native":
        raise ValueError("training-data override is native-only")
    if inputs.data_format == "native" and not allow_training_data:
        for split in getattr(backend, "training_splits", {}).values():
            if overlaps_provenance(inputs.rows, split):
                raise ValueError(
                    "evaluation overlaps head training/validation data; "
                    "use held-out data or explicitly --allow-training-data"
                )
    groups = defaultdict(list)
    for item in inputs.rows:
        if isinstance(item, LabeledDecision):
            context, question = item.context, item.question
            row = {
                "id": item.id,
                "family": question.type,
                "group_id": item.group_id,
                "label": item.gold_index,
                "options": [{"id": key} for key, _ in alternatives(question)],
            }
        else:
            context, question = ContextSpec(state=item.state), item.as_question()
            row = item.model_dump()
        groups[canonical_json(context.model_dump())].append((row, question))
    return groups


def _score_group(runtime, context_json, group, mode, projection):
    """Run one shared context; deletion in finally also releases it after a backend failure."""
    import json

    started = time.perf_counter()
    spec = ContextSpec.model_validate(json.loads(context_json))
    compiled = runtime.compile_context(spec)
    results, timings = [], []
    try:
        for start in range(0, len(group), 64):
            batch = group[start : start + 64]
            output = runtime.decide(
                DecisionRequest(
                    snapshot_id=compiled.snapshot.id,
                    namespace=spec.namespace,
                    questions={row["id"]: question for row, question in batch},
                    mode=mode,
                    projection=projection,
                )
            )
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
    finally:
        runtime.delete_context(compiled.snapshot.id, spec.namespace)
    timings[-1]["state_wall_ms"] = (time.perf_counter() - started) * 1000
    return results, timings


def evaluate(
    backend,
    source: Path | EvaluationInput,
    mode="cached",
    projection="auto",
    *,
    data_format="semif",
    allow_training_data=False,
) -> dict:
    """Run all frozen rows, grouping identical states while keeping per-row gold provenance."""
    inputs = (
        source if isinstance(source, EvaluationInput) else prepare_evaluation(source, data_format)
    )
    if inputs.data_format != data_format:
        raise ValueError("prepared evaluation format does not match requested format")
    groups = _evaluation_groups(backend, inputs, allow_training_data)
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
        predictions, measurements = _score_group(runtime, context_json, group, mode, projection)
        results.extend(predictions)
        timings.extend(measurements)
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
        "input_name": inputs.source.path.name,
        "input_sha256": inputs.source.sha256,
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
