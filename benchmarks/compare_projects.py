"""Run pinned upstream scorers and identical-prompt comparisons on Apple Silicon.

Use the environment installed from SemIf's pinned test/mlx extras plus Busbar.
No source patching, teacher calls, fitting, silent truncation or overwritten results.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from busbar.backends import load_backend
from busbar.compiler import canonical_json
from busbar.evaluation import metrics


def read_rows(path):
    """Preserve input order and fail on duplicate IDs before executing a model."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("nonempty input with unique IDs required")
    return rows


def revision(root):
    """Record the actual scorer checkout, refusing locally modified source."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("upstream source checkout must be clean")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def prediction(row, probabilities, logits=None, **details):
    """Map a complete distribution back to original option IDs without using gold in inference."""
    ids = [option["id"] for option in row["options"]]
    if (
        len(ids) != len(probabilities)
        or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        or abs(sum(probabilities) - 1) > 1e-5
    ):
        raise ValueError("incomplete candidate probabilities")
    selected = ids[max(range(len(ids)), key=probabilities.__getitem__)]
    return {
        "id": row["id"],
        "family": row["family"],
        "group_id": row["group_id"],
        "gold_index": row["label"],
        "gold_id": ids[row["label"]],
        "option_ids": ids,
        "selected": selected,
        "correct": selected == ids[row["label"]],
        "probabilities": probabilities,
        "logits": logits,
        "provenance": row.get("provenance", {}),
        **details,
    }


def summarize(records):
    """Use the same denominators and metrics as Busbar's own quality evaluation."""
    families = defaultdict(list)
    for row in records:
        families[row["family"]].append(row)
    by_family = {key: metrics(value) for key, value in families.items()}
    result = metrics(records)
    result["mean_family_balanced_accuracy"] = statistics.mean(
        value["balanced_accuracy"] for value in by_family.values()
    )
    return result, by_family


def busbar_same_prompt(backend, rows, prefix=None, projection="selected"):
    """Score SemIf's exact full token sequences using Busbar's production backend."""
    from semif_phase1.core import softmax
    from semif_phase1.direct import encode_prompt
    from semif_phase1.shared import _state_prefix

    encoded = [encode_prompt(backend.tokenizer, row, backend.max_tokens) for row in rows]
    prefix_ids = _state_prefix(backend.tokenizer, rows[0]["state"])
    if any(ids[: len(prefix_ids)] != prefix_ids for ids, _, _ in encoded):
        raise ValueError("shared prompt token prefix mismatch")
    if prefix is None:
        prefix = backend.prefill(tuple(prefix_ids))
    result = backend.score(
        [tuple(ids[len(prefix_ids) :]) for ids, _, _ in encoded],
        [tuple(slots) for _, slots, _ in encoded],
        prefix,
        projection,
    )
    return [
        {
            "id": row["id"],
            "option_ids": [o["id"] for o in row["options"]],
            "probabilities": softmax(scores),
            "option_logits": scores,
            "prompt_sha256": encoded_row[2],
            "input_tokens": len(encoded_row[0]),
        }
        for row, encoded_row, scores in zip(rows, encoded, result.logits, strict=True)
    ]


def quality(args, rows):
    """Run all rows through an unchanged upstream scorer or the controlled Busbar path."""
    import mlx.core as mx
    from semif_phase1 import mlx_backend as semif

    backend = None
    if args.project == "semif":
        model, tokenizer, identity = semif.load_model(args.model, args.revision)

        def run(group):
            """Keep SemIf's published fresh quality scorer unchanged."""
            return [semif.score(model, tokenizer, row, identity) for row in group]
    else:
        mx.set_cache_limit(256 * 1024 * 1024)
        backend = load_backend("mlx", args.model, args.revision, dtype="bfloat16", batch_size=32)
        identity = backend.identity

        def run(group):
            """Use exactly the same prompt tokens for the controlled comparison."""
            return busbar_same_prompt(backend, group)

    run(rows[:1])
    groups = defaultdict(list)
    for row in rows:
        groups[canonical_json(row["state"])].append(row)
    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter()
    records = []
    for group in groups.values():
        for start in range(0, len(group), 32):
            part = group[start : start + 32]
            outputs = run(part)
            for row, output in zip(part, outputs, strict=True):
                if output["id"] != row["id"] or output["option_ids"] != [
                    o["id"] for o in row["options"]
                ]:
                    raise ValueError("upstream scorer changed IDs or option order")
                records.append(
                    prediction(
                        row,
                        output["probabilities"],
                        output["option_logits"],
                        prompt_sha256=output["prompt_sha256"],
                        input_tokens=output["input_tokens"],
                    )
                )
        if len(records) % 16 == 0:
            print(f"{args.project}: {len(records)}/{len(rows)}", file=sys.stderr, flush=True)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    summary, families = summarize(records)
    summary.update(wall_seconds=elapsed, peak_mlx_active_bytes=mx.get_peak_memory())
    return {"model": identity, "summary": summary, "families": families, "predictions": records}


def speed(args, rows):
    """Compare all paths on one frozen 21-question state, with identical prompts and weights."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from semif_phase1 import mlx_backend as semif
    from semif_phase1.shared import _state_prefix

    first_group = rows[0]["group_id"]
    rows = [row for row in rows if row["group_id"] == first_group]
    if len(rows) != 21 or any(row["state"] != rows[0]["state"] for row in rows):
        raise ValueError("expected the complete first 21-question shared-state group")
    mx.set_cache_limit(256 * 1024 * 1024)
    backend = load_backend("mlx", args.model, args.revision, dtype="bfloat16", batch_size=32)
    model, tokenizer, identity = backend.model, backend.tokenizer, backend.identity

    def serial():
        """Include SemIf's one state prefill and all subsequent serial suffix calls."""
        scorer = semif.SerialPrefixScorer(model, tokenizer, identity)
        return [scorer.score(row) for row in rows]

    warm_serial = semif.SerialPrefixScorer(model, tokenizer, identity)
    warm_serial.score(rows[0])
    prefix = backend.prefill(tuple(_state_prefix(tokenizer, rows[0]["state"])))
    methods = {
        "semif_fresh_serial": lambda: [
            semif.score(model, tokenizer, row, identity) for row in rows
        ],
        "semif_cold_serial": serial,
        "semif_cold_shared": lambda: semif.score_shared(model, tokenizer, rows, identity)[0],
        "semif_warm_serial": lambda: [warm_serial.score(row) for row in rows],
        "busbar_cold_shared": lambda: busbar_same_prompt(backend, rows),
        "busbar_warm_shared": lambda: busbar_same_prompt(backend, rows, prefix),
        "busbar_warm_full_head": lambda: busbar_same_prompt(backend, rows, prefix, "full"),
    }
    for name, method in methods.items():
        method()
        print(f"warmed {name}", file=sys.stderr, flush=True)
    runs = {name: [] for name in methods}
    rng = random.Random(217)
    for iteration in range(args.repeats):
        names = list(methods)
        rng.shuffle(names)
        for name in names:
            mx.synchronize()
            started = time.perf_counter()
            output = methods[name]()
            mx.synchronize()
            elapsed = time.perf_counter() - started
            runs[name].append({"seconds": elapsed, "outputs": output})
            print(
                f"{iteration + 1}/{args.repeats} {name}: {elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )
    return {
        "model": identity,
        "dtype_counts": dict(_dtype_counts(tree_flatten(model.parameters()))),
        "row_ids": [row["id"] for row in rows],
        "repeats": args.repeats,
        "busbar_batch_limit": backend.batch_size,
        "retained_prefix_bytes": prefix.nbytes,
        "summary": {
            name: {
                "median_seconds": statistics.median(r["seconds"] for r in values),
                "samples_seconds": [r["seconds"] for r in values],
            }
            for name, values in runs.items()
        },
        "runs": runs,
        "scope": "Same resident weights, BF16, SemIf prompts, 21 original rows; encoding included. "
        "Cold paths include one prefill; warm paths exclude it. No question KV cache. "
        "Models run sequentially. SemIf uses its unchanged production scorer functions.",
    }


def _dtype_counts(tensors):
    """Report actual loaded tensors, including architecture-specific FP32 state parameters."""
    counts = defaultdict(int)
    for _, value in tensors:
        counts[str(value.dtype)] += value.size
    return counts


def nano_quality(args, rows):
    """Port only loading/device orchestration; execute original NanoJev model and token builder."""
    import torch
    from predict_toy_decisions import load_decision_model_class, prepare_examples
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    root = args.checkpoint
    config = json.loads((root / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    body_config = AutoConfig.from_pretrained(
        root / "backbone_config", local_files_only=True, trust_remote_code=False
    )
    body = AutoModel.from_config(
        body_config, attn_implementation="sdpa", trust_remote_code=False
    ).float()
    model = load_decision_model_class()(body, config["set_head"])
    weights = load_file(root / "best.safetensors", device="cpu")
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device(args.device)
    model.to(device=device, dtype=torch.float32).eval()
    sync = torch.mps.synchronize if device.type == "mps" else lambda: None
    prepared, records = [], []
    for row in rows:
        payload = {
            "states": [
                {
                    "id": row["id"],
                    "state": row["state"],
                    "questions": {
                        "decision": {
                            "type": "choice",
                            "instructions": row["question"],
                            "criteria": {o["id"]: o["description"] for o in row["options"]},
                        }
                    },
                }
            ]
        }
        try:
            example = prepare_examples(payload, tokenizer, config.get("max_length", 512))[0]
            prepared.append((row, example))
        except ValueError as exc:
            records.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "gold_id": row["options"][row["label"]]["id"],
                    "correct": False,
                    "error": str(exc),
                }
            )
    if not prepared:
        raise ValueError("NanoJev accepted no rows")
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.nano_autocast),
    ):
        model([prepared[0][1]], tokenizer.pad_token_id)
        sync()
        started = time.perf_counter()
        for start in range(0, len(prepared), 4):
            batch = prepared[start : start + 4]
            scores, _ = model([ex for _, ex in batch], tokenizer.pad_token_id)
            for (row, example), values in zip(batch, scores, strict=True):
                logits = values[: len(example["candidate_ids"])].float()
                records.append(
                    prediction(
                        row,
                        logits.softmax(-1).cpu().tolist(),
                        logits.cpu().tolist(),
                        candidate_path_tokens=[len(p) for p in example["leaf_tokens"]],
                    )
                )
            if len(records) % 16 == 0:
                print(f"nanojev: {len(records)}/{len(rows)}", file=sys.stderr, flush=True)
        sync()
    summary, families = summarize(records)
    summary["wall_seconds"] = time.perf_counter() - started
    return {
        "model": {
            "source": "C-Tianyu/NanoJev",
            "revision": args.revision,
            "device": str(device),
            "parameter_storage": "float32",
            "dtype": "bfloat16_autocast" if args.nano_autocast else "float32",
            "base_model": config.get("model"),
            "set_head": config["set_head"],
            "checkpoint_max_tokens": config.get("max_length", 512),
            "batch_questions": 4,
        },
        "summary": summary,
        "families": families,
        "predictions": records,
        "scope": "Original trained root checkpoint and original model/encoding functions. "
        "Mac device port, not official CUDA service timing. No retraining. "
        "Failures remain in the denominator. This is out-of-domain quality evaluation.",
    }


def main():
    """Require explicit sources and create-only outputs so each comparison can be audited."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", choices=("semif", "busbar-semif-prompt", "nanojev"), required=True
    )
    parser.add_argument("--task", choices=("quality", "speed"), default="quality")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--nano-autocast", action="store_true", help="NanoJev BF16 forward autocast"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be 1-10")
    root = args.source_root.resolve()
    source_revision = revision(root)
    sys.path.insert(0, str(root / ("scripts" if args.project == "nanojev" else "src")))
    rows = read_rows(args.input)
    if args.project == "nanojev":
        if args.task != "quality" or args.checkpoint is None:
            parser.error("NanoJev requires --checkpoint and quality task")
        report = nano_quality(args, rows)
    else:
        report = speed(args, rows) if args.task == "speed" else quality(args, rows)
    versions = {}
    for name in ("mlx", "mlx-lm", "torch", "transformers", "busbar-runtime", "semif-phase1"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report.update(
        schema_version=1,
        project=args.project,
        task=args.task,
        upstream_revision=source_revision,
        input_name=args.input.name,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        versions=versions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
