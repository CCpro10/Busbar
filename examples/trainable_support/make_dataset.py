"""Create a small, authored support-triage fixture; split utterances before deriving labels."""

import json
import random
from pathlib import Path

INTENTS = {
    "delivery": (
        "Package tracking and delivery",
        [
            "Where is my parcel?",
            "My order has not arrived.",
            "Track the shipment for me.",
            "The courier went to the wrong address.",
            "When will the package arrive?",
            "The tracking link is broken.",
            "My shipment is delayed.",
            "I missed the courier.",
        ],
        [
            "The parcel says delivered but I cannot find it.",
            "Can I change the delivery address?",
            "Which carrier is bringing my order?",
        ],
        [
            "My box has been stuck at the depot.",
            "The delivery driver never called.",
            "Can the courier leave the parcel downstairs?",
            "Is my order already in transit?",
        ],
    ),
    "returns": (
        "Product returns, exchanges and refunds",
        [
            "I want to return these shoes.",
            "Can I get a refund for this item?",
            "The jacket does not fit; I want to send it back.",
            "Please exchange this shirt.",
            "How do I return a damaged product?",
            "I need a return label.",
            "The item was broken; please refund it.",
            "I changed my mind about this purchase.",
        ],
        [
            "Can I swap this for another size?",
            "I sent the product back; where is my refund?",
            "This purchase is defective and I want my money back.",
        ],
        [
            "The trousers are too small and need exchanging.",
            "Please reimburse my returned item.",
            "I would like to cancel the purchase and send it back.",
            "The wrong product came; how do I exchange it?",
        ],
    ),
    "billing": (
        "Payment errors, invoices and duplicate charges",
        [
            "I was charged twice.",
            "Please send my invoice.",
            "My payment was declined.",
            "The bill has an incorrect amount.",
            "Why did you charge my card again?",
            "I need a receipt for my payment.",
            "I cannot complete the payment.",
            "Please correct the tax on my invoice.",
        ],
        [
            "There are two identical card transactions.",
            "My invoice has the wrong company name.",
            "The checkout keeps rejecting my debit card.",
        ],
        [
            "The amount on the receipt is not what I paid.",
            "Can you issue a VAT invoice?",
            "I paid once but my bank shows two deductions.",
            "The payment screen reports an error.",
        ],
    ),
    "account": (
        "Login, passwords and account access",
        [
            "I forgot my password.",
            "I cannot log in.",
            "Please unlock my account.",
            "The password reset email never arrived.",
            "My login code is invalid.",
            "I need to change my account email.",
            "My account has been locked.",
            "Someone changed my login password.",
        ],
        [
            "The sign-in page does not recognize my username.",
            "How can I recover my account?",
            "My two-factor code does not work.",
        ],
        [
            "I lost access to the email used for signing in.",
            "My reset link has expired.",
            "I am unable to authenticate with my security code.",
            "Please help me regain login access.",
        ],
    ),
}
URGENCY = {
    "train": [
        "There is no rush.",
        "Please help me today.",
        "This is an emergency; help immediately.",
    ],
    "validation": [
        "Whenever you have time is fine.",
        "I need this resolved by tonight.",
        "This is critical and cannot wait.",
    ],
    "test": [
        "It can wait until next week.",
        "Please sort this out before the day ends.",
        "I need urgent assistance right now.",
    ],
}


def build(split, seed):
    """Group all three questions from a ticket together; shuffle and vary choice sets."""
    rng = random.Random(seed)
    rows = []
    for intent_index, (intent, parts) in enumerate(INTENTS.items()):
        for index, message in enumerate(parts[{"train": 1, "validation": 2, "test": 3}[split]]):
            group = f"{split}-{intent}-{index}"
            urgency = (index + intent_index) % 3
            context = {
                "namespace": "support-demo",
                "state": {"message": message + " " + URGENCY[split][urgency]},
                "instructions": (
                    "Classify the customer message. Use its meaning, not any option ID."
                ),
            }
            other = [key for key in INTENTS if key != intent]
            candidates = [intent] + rng.sample(other, 1 + index % 3)
            rng.shuffle(candidates)
            questions = {
                "route": (
                    {
                        "type": "choice",
                        "question": "Which support team should handle this?",
                        "options": [
                            {"id": key, "description": INTENTS[key][0]} for key in candidates
                        ],
                    },
                    intent,
                ),
                "money": (
                    {
                        "type": "boolean",
                        "question": "Is this about a product return/refund or a payment/invoice?",
                    },
                    intent in ("returns", "billing"),
                ),
                "urgency": (
                    {
                        "type": "score",
                        "question": "How urgent is the requested help?",
                        "levels": [
                            {"value": 0, "description": "Low: can wait; no rush"},
                            {"value": 1, "description": "Medium: needs help today"},
                            {"value": 2, "description": "High: immediate emergency"},
                        ],
                    },
                    urgency,
                ),
            }
            for name, (question, label) in questions.items():
                rows.append(
                    {
                        "id": f"{group}-{name}",
                        "group_id": group,
                        "context": context,
                        "question": question,
                        "label": label,
                    }
                )
    return rows


def main():
    """Regenerate deterministic public fixtures; this script never observes model predictions."""
    for seed, split in enumerate(("train", "validation", "test"), 42):
        path = Path(__file__).parent / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build(split, seed))
        )
    # External scorers expose a categorical contract. Preserve candidate semantics,
    # gold indices and state instructions while allowing each project its own prompt.
    from busbar.compiler import alternatives, canonical_json
    from busbar.datasets import read_dataset

    root = Path(__file__).parent
    converted = []
    for row in read_dataset(root / "test.jsonl"):
        converted.append(
            {
                "id": row.id,
                "group_id": row.group_id,
                "family": row.question.type,
                "state": canonical_json(row.context.model_dump(exclude={"namespace"})),
                "question": row.question.question,
                "options": [
                    {"id": key, "description": text} for key, text in alternatives(row.question)
                ],
                "label": row.gold_index,
                "provenance": {
                    "source": "authored Busbar support demo; 16 held-out tickets",
                    "mapping": "all types evaluated as categorical decisions",
                },
            }
        )
    (root / "test-categorical.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted)
    )


if __name__ == "__main__":
    main()
