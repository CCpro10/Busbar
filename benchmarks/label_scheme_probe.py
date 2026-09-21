"""Measure how a label alphabet affects vocabulary-readout accuracy on wide choice questions.

Raising the Choice ceiling needs an alphabet of single-token labels, but tokenization alone does
not settle it: the labels also sit in the prompt, and a model that saw "A. / B. / C." throughout
training may read an unfamiliar symbol alphabet worse, or hold a different prior over the label
tokens themselves. This probe runs real benchmark items through the production backend under
several alphabets and reports accuracy plus a label-concentration diagnostic, so a ceiling change
is decided on measurements rather than on a token count.

The probe changes no production code: it builds the suffix the compiler would build for a given
alphabet, and reads the same label rows through the same backend.
"""

import argparse
import collections
import json
import random
import statistics
import time
from pathlib import Path

from busbar.backends import load_backend
from busbar.compiler import SYSTEM, canonical_json

# Candidate alphabets. Each must supply enough single-token entries for the widest question tested;
# entries are verified against the loaded tokenizer before use, never assumed.
ALPHABETS = {
    # Today's production alphabet, extended through the rest of the uppercase letters.
    "upper": [chr(ord("A") + i) for i in range(26)],
    # Uppercase then lowercase: familiar glyphs, but "A" and "a" differ only by case.
    "mixed_case": [chr(ord("A") + i) for i in range(26)] + [chr(ord("a") + i) for i in range(26)],
    # Spreadsheet-style pairs: unambiguous and familiar from column headers.
    "pairs": [f"{a}{b}" for a in "ABCDEFGH" for b in "ABCDEFGHIJKLMNOP"],
    # The user's proposal: single-token special characters.
    "symbols": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + list("¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿×÷"),
    # Greek and Cyrillic letters: single glyphs, still letter-like.
    "greek_cyrillic": [chr(c) for c in range(0x391, 0x3CA)] + [chr(c) for c in range(0x410, 0x450)],
}


def single_token(tokenizer, text: str) -> bool:
    """A label must round-trip as exactly one token to be readable from one hidden state."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids) == 1 and tokenizer.decode(ids) == text


def usable(tokenizer, alphabet: list[str], width: int) -> list[str]:
    """Keep the alphabet's own order; drop entries the tokenizer splits."""
    kept = [entry for entry in alphabet if single_token(tokenizer, entry)]
    if len(kept) < width:
        raise ValueError(f"alphabet supplies {len(kept)} single-token labels, need {width}")
    return kept[:width]


def read_items(path: Path, task: str, limit: int) -> list[dict]:
    """Frozen benchmark items for one task, in file order, skipping ids-only rows."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("meta") or {}).get("task") != task:
            continue
        if not row.get("state") or not row.get("questions") or not row.get("gold"):
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def scored_question(item: dict) -> tuple[str, dict, str]:
    """The item's primary gold choice question, which is what this probe scores."""
    for qid, gold in item["gold"].items():
        question = item["questions"][qid]
        if question["type"] == "choice":
            return qid, question, str(gold)
    raise ValueError("item has no gold choice question")


def build_suffix(backend, question: dict, order: list[str], labels: list[str]):
    """Render the decision suffix exactly as the production compiler does, for a given alphabet."""
    criteria = question["criteria"]
    choices = "\n".join(f"{labels[i]}. {criteria[key]}" for i, key in enumerate(order))
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}]
    prefix = backend.tokenizer.apply_chat_template(
        base, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    full = backend.tokenizer.apply_chat_template(
        base
        + [
            {
                "role": "user",
                "content": f"Question: {question['instructions']}\nAlternatives:\n{choices}",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not full.startswith(prefix):
        raise ValueError("unsupported chat template: adding a question rewrites the prefix")
    text = full[len(prefix) :]
    ids = backend.tokenizer.encode(text, add_special_tokens=False)
    label_ids = []
    for label in labels:
        slot = backend.tokenizer.encode(label, add_special_tokens=False)
        if backend.tokenizer.encode(text + label, add_special_tokens=False) != ids + slot:
            raise ValueError(f"label {label!r} changes the answer token boundary")
        label_ids.append(slot[0])
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("label token IDs collide")
    return tuple(ids), tuple(label_ids)


def context_tokens(backend, item: dict) -> tuple[int, ...]:
    """The cached context message, identical across alphabets for one item."""
    messages = [
        {"role": "system", "content": SYSTEM + "\n"},
        {"role": "user", "content": "Context data:\n" + canonical_json(item["state"])},
    ]
    text = backend.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    return tuple(backend.tokenizer.encode(text, add_special_tokens=False))


def run_alphabet(backend, items: list[dict], alphabet: list[str], seed: int) -> dict:
    """Score every item once per alphabet, with a seeded option order shared across alphabets."""
    records = []
    started = time.perf_counter()
    for item in items:
        qid, question, gold = scored_question(item)
        order = sorted(question["criteria"])
        random.Random(seed).shuffle(order)
        labels = usable(backend.tokenizer, alphabet, len(order))
        suffix, label_ids = build_suffix(backend, question, order, labels)
        prefix = backend.prefill(context_tokens(backend, item))
        result = backend.score([suffix], [label_ids], prefix, "selected")
        logits = result.logits[0]
        best = max(range(len(logits)), key=logits.__getitem__)
        records.append(
            {
                "id": item["id"],
                "options": len(order),
                "gold": gold,
                "selected": order[best],
                "correct": order[best] == gold,
                "selected_label_index": best,
            }
        )
    return {
        "items": len(records),
        "accuracy": round(statistics.mean(r["correct"] for r in records), 4),
        "wall_seconds": round(time.perf_counter() - started, 2),
        # How concentrated the picks are on a few label positions: a healthy readout spreads across
        # labels roughly as the gold answers do, while a biased alphabet piles onto a few tokens.
        "top_label_share": round(
            collections.Counter(r["selected_label_index"] for r in records).most_common(1)[0][1]
            / len(records),
            4,
        ),
        "distinct_labels_used": len({r["selected_label_index"] for r in records}),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True, help="frozen suite items.jsonl")
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--alphabet", action="append", choices=sorted(ALPHABETS), default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")

    items = read_items(args.items, args.task, args.limit)
    if not items:
        parser.error("no usable items for that task")
    width = max(len(scored_question(item)[1]["criteria"]) for item in items)
    backend = load_backend("mlx", args.model, args.revision, dtype=args.dtype, batch_size=8)

    names = args.alphabet or sorted(ALPHABETS)
    report = {
        "schema_version": 1,
        "task": args.task,
        "items": len(items),
        "widest_question": width,
        "model": backend.identity,
        "option_order_seed": args.seed,
        "alphabets": {},
    }
    for name in names:
        try:
            run = run_alphabet(backend, items, ALPHABETS[name], args.seed)
        except ValueError as exc:
            report["alphabets"][name] = {"unavailable": str(exc)}
            print(f"{name}: unavailable - {exc}")
            continue
        report["alphabets"][name] = run
        print(
            f"{name}: accuracy {run['accuracy']} | top-label share {run['top_label_share']} | "
            f"{run['distinct_labels_used']} distinct labels | {run['wall_seconds']}s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
