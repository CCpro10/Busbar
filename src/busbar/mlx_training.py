"""Frozen-backbone feature extraction and small-head optimization on MLX.

All accelerator imports are local so dataset validation and CLI help also work on Linux.
"""

import math
import random
import sys

from .compiler import CandidateCompiler


def extract_features(backend, rows, feature_cache_mib):
    """Preflight every path, then encode with a frozen body and disposable KV branches."""
    import mlx.core as mx

    compiler = CandidateCompiler(backend.tokenizer)
    size = backend.model.model.embed_tokens.weight.shape[1]
    compiled = [(compiler.context(row.context), compiler.question(row.question)) for row in rows]
    max_candidates = max(len(question.paths) for _, question in compiled)
    # Include both the per-row arrays and their stacked copy; model/KV memory is separate.
    required = 2 * len(rows) * max_candidates * size * 4
    if required > feature_cache_mib * 1024 * 1024:
        raise ValueError(f"feature cache needs at least {math.ceil(required / 2**20)} MiB")
    if any(
        len(prefix) + len(path.token_ids) > backend.max_tokens
        for prefix, question in compiled
        for path in question.paths
    ):
        raise ValueError("training input exceeds token limit; no examples were truncated")
    backend.model.freeze()
    features, masks = [None] * len(rows), [None] * len(rows)
    groups = {}
    for index, (prefix, _) in enumerate(compiled):
        groups.setdefault(prefix, []).append(index)
    completed = 0
    for prefix_tokens, indices in groups.items():
        prefix = backend.prefill(prefix_tokens)
        for index in indices:
            paths = compiled[index][1].paths
            vectors = [None] * len(paths)
            for positions, hidden in backend.hidden_batches([p.token_ids for p in paths], prefix):
                hidden = mx.stop_gradient(hidden.astype(mx.float32))
                mx.eval(hidden)
                for position, vector in zip(positions, hidden, strict=True):
                    vectors[position] = vector
            features[index] = mx.stack(
                vectors + [mx.zeros((size,))] * (max_candidates - len(paths))
            )
            masks[index] = [i < len(paths) for i in range(max_candidates)]
            mx.eval(features[index])
            completed += 1
            if completed % 24 == 0 or completed == len(rows):
                print(f"features {completed}/{len(rows)}", file=sys.stderr, flush=True)
        del prefix
    result = (
        mx.stack(features),
        mx.array(masks),
        mx.array([row.gold_index for row in rows]),
    )
    mx.eval(result)
    return result


def _logits(head, features, mask):
    """Exclude padding from both the training objective and quality measurements."""
    import mlx.core as mx

    return mx.where(mask, head(features), -1e9)


def feature_metrics(head, data):
    """Measure categorical accuracy and NLL without fitting calibration on validation examples."""
    import mlx.core as mx
    import mlx.nn as nn

    features, mask, labels = data
    logits = _logits(head, features, mask)
    nll = float(nn.losses.cross_entropy(logits, labels, reduction="mean").item())
    accuracy = float(mx.mean(mx.argmax(logits, axis=-1) == labels).item())
    if not math.isfinite(nll):
        raise ValueError("nonfinite training metrics; artifact was not published")
    return {"rows": len(labels), "accuracy": accuracy, "nll": nll}


def fit_head(head, train, validation, config):
    """Optimize only the head; choose a checkpoint by validation NLL, including epoch zero."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    def loss(module, features, mask, labels):
        """One mutually exclusive categorical target per variable-length candidate set."""
        return nn.losses.cross_entropy(_logits(module, features, mask), labels, reduction="mean")

    gradient = nn.value_and_grad(head, loss)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    shuffle = random.Random(config.seed)
    initial = dict(tree_flatten(head.parameters()))
    history = [
        {
            "epoch": 0,
            "train": feature_metrics(head, train),
            "validation": feature_metrics(head, validation),
        }
    ]
    best, best_epoch, best_loss = initial, 0, history[0]["validation"]["nll"]
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train[2])))
        shuffle.shuffle(order)
        for start in range(0, len(order), config.train_batch_size):
            indices = mx.array(order[start : start + config.train_batch_size])
            value, grads = gradient(head, *(item[indices] for item in train))
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, value)
            if not math.isfinite(float(value.item())):
                raise ValueError("nonfinite training loss; artifact was not published")
        record = {
            "epoch": epoch,
            "train": feature_metrics(head, train),
            "validation": feature_metrics(head, validation),
        }
        history.append(record)
        if record["validation"]["nll"] < best_loss:
            best = dict(tree_flatten(head.parameters()))
            best_epoch, best_loss = epoch, record["validation"]["nll"]
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch {epoch}: train NLL={record['train']['nll']:.4f}, "
                f"validation NLL={record['validation']['nll']:.4f}",
                file=sys.stderr,
                flush=True,
            )
    head.load_weights(list(best.items()), strict=True)
    head.eval()
    delta = sum(float(mx.sum((best[key] - value) ** 2).item()) for key, value in initial.items())
    return {
        "history": history,
        "best_epoch": best_epoch,
        "parameter_delta_l2": math.sqrt(delta),
        "trainable_parameters": sum(value.size for value in best.values()),
        "selected_train": feature_metrics(head, train),
        "selected_validation": feature_metrics(head, validation),
    }
