"""Recompute published comparisons from complete, paired raw predictions."""

import argparse
import json
from pathlib import Path

from busbar.evaluation import metrics


def paired(left, right):
    """Reject changed fixtures/prompts before reporting agreement or probability drift."""
    assert left["input_sha256"] == right["input_sha256"]
    assert left["model"]["revision"] == right["model"]["revision"]
    a = {row["id"]: row for row in left["predictions"]}
    b = {row["id"]: row for row in right["predictions"]}
    assert len(a) == len(left["predictions"]) and len(b) == len(right["predictions"])
    assert a.keys() == b.keys()
    for key in a:
        for field in ("gold_id", "option_ids", "prompt_sha256", "input_tokens"):
            assert a[key][field] == b[key][field], (key, field)
    return {
        "rows": len(a),
        "argmax_disagreements": sum(a[key]["selected"] != b[key]["selected"] for key in a),
        "max_probability_delta": max(
            abs(x - y)
            for key in a
            for x, y in zip(a[key]["probabilities"], b[key]["probabilities"], strict=True)
        ),
        "max_logit_delta": max(
            abs(x - y) for key in a for x, y in zip(a[key]["logits"], b[key]["logits"], strict=True)
        ),
        "disagreement_ids": [key for key in a if a[key]["selected"] != b[key]["selected"]],
    }


def summarize(root):
    """Keep model/prompt comparisons separate from controlled implementation timing."""

    def read(name):
        """Require the full named artifact instead of silently omitting missing runs."""
        return json.loads((root / f"{name}.json").read_text())

    prefixes = (
        "busbar-06b",
        "busbar-qwen25",
        "busbar-minicpm5",
        "busbar-qwen35",
        "semif",
        "busbar-semif-prompt",
        "nanojev-bf16",
        "metal-minicpm5",
    )
    quality = {}
    for prefix in prefixes:
        reports = [read(f"{prefix}-{dataset}") for dataset in ("authored144", "wanli256")]
        assert [r["summary"]["rows"] for r in reports] == [144, 256]
        assert all(r["summary"]["coverage"] == 1 for r in reports)
        quality[prefix] = {
            "authored144": reports[0]["summary"],
            "wanli256": reports[1]["summary"],
            "combined400": metrics([row for r in reports for row in r["predictions"]]),
        }
    controlled = {
        dataset: paired(read(f"semif-{dataset}"), read(f"busbar-semif-prompt-{dataset}"))
        for dataset in ("authored144", "wanli256")
    }
    speed = read("speed-qwen35-shape21")
    reference = speed["runs"]["semif_fresh_serial"][0]["outputs"]
    speed_agreement = {}
    for method, runs in speed["runs"].items():
        flips, delta = [], 0.0
        for run in runs:
            assert len(run["outputs"]) == len(reference) == 21
            changed = 0
            for row, ref in zip(run["outputs"], reference, strict=True):
                for field in ("id", "option_ids", "prompt_sha256", "input_tokens"):
                    assert row[field] == ref[field], (method, field)
                p, q = row["probabilities"], ref["probabilities"]
                changed += max(range(len(p)), key=p.__getitem__) != max(
                    range(len(q)), key=q.__getitem__
                )
                delta = max(delta, *(abs(x - y) for x, y in zip(p, q, strict=True)))
            flips.append(changed)
        speed_agreement[method] = {"argmax_disagreements_per_run": flips, "max_prob_delta": delta}
    return {
        "quality": quality,
        "same_prompt_quality": controlled,
        "same_prompt_speed": speed["summary"],
        "speed_agreement_with_fresh_semif": speed_agreement,
        "scope": "Combined400 is this study's unweighted aggregate, not an official leaderboard. "
        "BF16 results may differ near ties. Speed covers one frozen 21-question state, three runs.",
    }


def main():
    """Write a create-only artifact while retaining independently auditable source predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = summarize(args.root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
